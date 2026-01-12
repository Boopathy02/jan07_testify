import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traceback
import asyncio
import subprocess
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text

try:
    from psycopg.errors import UndefinedTable as PsycopgUndefinedTable
except Exception:  # pragma: no cover - optional import guard
    PsycopgUndefinedTable = None

# Routers
from apis.image_text_api import router as image_router
from apis.chroma_debug_api import router as debug_chroma_export_router
from apis.enrichment_api import router as enrichment_router
from apis.url_enrichment import router as url_enrichment_router
from apis.rag_testcase_runner import router as rag_router
from apis.generate_from_story import router as generate_from_story_router
from apis.generate_page_methods import router as generate_page_methods_router
from apis.generate_from_manual_testcases import router as generate_from_manual_testcase_router
from apis.generate_testcases_from_methods import router as generate_test_code_from_methods_router
from apis.manual_add_metadata import router as manual_add_metadata
from apis.manual_enrichment_api import router as manual_enrichment_router  # from first file
from apis.projects_api import router as projects_router
from apis.run_test_api import router as run_tests_router
from apis.report_api import router as report_router
from apis.markers_api import router as markers_router  # from second file
from apis.metrics_api import router as metrics_router  # from second file
from apis.jira_api import router as jira_router

import auth
from database.models import Organization, User
from database.session import engine, get_db
from database.migration_runner import auto_migrate_enabled, run_migrations_if_needed
from utils.security import hash_password, verify_password


# -------------------------------------------------------
# DB STARTUP VALIDATION (NO RUNTIME DDL)
# -------------------------------------------------------
def _alembic_config():
    from alembic.config import Config

    backend_root = Path(__file__).resolve().parent
    alembic_ini = backend_root / "database" / "alembic.ini"
    migrations_dir = backend_root / "database" / "migrations"

    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(migrations_dir))
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        config.set_main_option("sqlalchemy.url", db_url)
    return config


def _required_tables_exist() -> bool:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    return {"organizations", "users", "projects"}.issubset(tables)


def _current_db_revision() -> str | None:
    inspector = inspect(engine)
    if "alembic_version" not in set(inspector.get_table_names()):
        return None
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version_num FROM alembic_version"))
        row = result.first()
        return row[0] if row else None


def _alembic_head_revision() -> str:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config())
    heads = script.get_heads()
    if not heads:
        raise RuntimeError("Alembic head revision not found.")
    if len(heads) > 1:
        raise RuntimeError(f"Multiple Alembic heads detected: {heads}")
    return heads[0]


_startup_logger = logging.getLogger("startup")


def _sanitize_database_url(url: str | None) -> str:
    if not url:
        return "<not set>"
    try:
        from sqlalchemy.engine import make_url

        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<invalid>"


def _validate_database_schema() -> None:
    required_missing = not _required_tables_exist()
    current = _current_db_revision()
    head = _alembic_head_revision()
    out_of_sync = required_missing or not current or current != head

    if out_of_sync and auto_migrate_enabled():
        run_migrations_if_needed(engine, _alembic_config(), _startup_logger)
        required_missing = not _required_tables_exist()
        current = _current_db_revision()
        head = _alembic_head_revision()
        out_of_sync = required_missing or not current or current != head

    if out_of_sync:
        db_url = _sanitize_database_url(os.getenv("DATABASE_URL"))
        _startup_logger.error(
            "Database schema out of sync. current=%s head=%s db=%s required_tables_missing=%s",
            current,
            head,
            db_url,
            required_missing,
        )
        raise RuntimeError("Database schema out of sync. Run Alembic migrations.")


# -------------------------------------------------------
# Playwright Windows Fix
# -------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        print("Playwright install failed:", e)


# -------------------------------------------------------
# FASTAPI INITIALIZATION
# -------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        _validate_database_schema()
    except RuntimeError:
        raise SystemExit(1)
    yield


app = FastAPI(title="AI Test Extractor", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# AUTH MODELS
# -------------------------------------------------------
class _AuthBase(BaseModel):
    organization: str
    email: EmailStr
    password: str

    @field_validator("organization")
    @classmethod
    def _organization_not_empty(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Organization is required.")
        return cleaned

    @field_validator("password")
    @classmethod
    def _password_min_length(cls, value: str) -> str:
        if not value or len(value) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return value

    def normalized_email(self) -> str:
        return str(self.email).strip().lower()

    def normalized_org(self) -> str:
        return (self.organization or "").strip().lower()


class SignupRequest(_AuthBase):
    pass


class LoginRequest(_AuthBase):
    pass


def _is_schema_missing_error(exc: Exception) -> bool:
    if PsycopgUndefinedTable and isinstance(getattr(exc, "orig", None), PsycopgUndefinedTable):
        return True
    message = str(exc).lower()
    return "relation" in message and "does not exist" in message


# -------------------------------------------------------
# SIGNUP
# -------------------------------------------------------
@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup_user(payload: SignupRequest, db: Session = Depends(get_db)):
    try:
        org = Organization.get_or_create(db, payload.organization)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (ProgrammingError, OperationalError) as exc:
        if _is_schema_missing_error(exc):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database schema out of sync. Run Alembic migrations.",
            ) from exc
        raise
    except Exception as exc:
        if _is_schema_missing_error(exc):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database schema out of sync. Run Alembic migrations.",
            ) from exc
        raise

    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # Defensive: ensure callers see a clean error.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process password securely.",
        ) from exc

    user = User(
        organization=org.display_name,
        organization_id=org.id,
        email=payload.normalized_email(),
        password_hash=password_hash,
    )

    try:
        db.add(user)
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )
    except (ProgrammingError, OperationalError) as exc:
        db.rollback()
        if _is_schema_missing_error(exc):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database schema out of sync. Run Alembic migrations.",
            ) from exc
        raise
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    db.refresh(user)
    return {"status": "created", "user": user.to_dict()}


# -------------------------------------------------------
# LOGIN
# -------------------------------------------------------
@app.post("/login")
def login_for_access_token(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.normalized_email()
    organization = payload.normalized_org()

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials.")

    stored_org = (user.organization or "").strip().lower()

    if stored_org != organization or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect credentials.")

    access_token = auth.create_access_token(
        data={"sub": user.email, "uid": user.id, "org": user.organization}
    )

    return {"access_token": access_token, "token_type": "bearer"}


# -------------------------------------------------------
# GLOBAL EXCEPTION HANDLER
# -------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("❌ Unhandled Exception:")
    traceback.print_exc()

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true",
    }

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers=headers,
    )


# -------------------------------------------------------
# ROUTERS
# -------------------------------------------------------
app.include_router(image_router)
app.include_router(generate_from_story_router)
app.include_router(enrichment_router)
app.include_router(url_enrichment_router)
app.include_router(rag_router)
app.include_router(debug_chroma_export_router)
app.include_router(generate_from_manual_testcase_router)
app.include_router(generate_page_methods_router)
app.include_router(generate_test_code_from_methods_router)
app.include_router(manual_add_metadata)

# merged additions from both files
app.include_router(manual_enrichment_router)        # from first file
app.include_router(projects_router)
app.include_router(markers_router)                 # from second file
app.include_router(run_tests_router, prefix="/tests")
app.include_router(report_router, prefix="/reports")
app.include_router(metrics_router, prefix="/metrics")  # from second file
app.include_router(jira_router)


# -------------------------------------------------------
# MAIN SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.ERROR)
    logging.getLogger("tqdm").setLevel(logging.WARNING)

    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False, log_level="info")
