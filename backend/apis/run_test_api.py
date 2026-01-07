import logging
import os
import subprocess
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from metrics.collector import collect_run_summary
from metrics.store import MetricsStore
from .report_api import _resolve_src_dir

router = APIRouter()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VISUALIZER_SCRIPT = REPO_ROOT / "allure_reports" / "allure_visualizer.py"


def _shorten_output(text: str, limit: int = 400) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}... (truncated)"


def _ensure_env(src_dir: Path, project_dir: Path, keep_browser: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(src_dir)]
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["SMARTAI_SRC_DIR"] = str(src_dir)
    env["SMARTAI_PROJECT_DIR"] = str(project_dir)
    os.environ["SMARTAI_SRC_DIR"] = str(src_dir)
    os.environ["SMARTAI_PROJECT_DIR"] = str(project_dir)
    hold_value = keep_browser or env.get("UI_KEEP_BROWSER_OPEN") or os.environ.get("UI_KEEP_BROWSER_OPEN")
    if not hold_value:
        hold_value = "30"  # give UI time to stay visible by default
    env["UI_KEEP_BROWSER_OPEN"] = hold_value
    env["SMARTAI_SKIP_PLAYWRIGHT_FIXTURES"] = "1"

    return env


def _run_pytest(test_file: Path, allure_results_dir: Path, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:playwright",
        str(test_file),
        "--alluredir",
        str(allure_results_dir),
    ]
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


def _generate_report(results_dir: Path, report_dir: Path, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    allure_exe = shutil.which("allure")
    if not allure_exe:
        raise HTTPException(status_code=501, detail="Allure CLI is not installed. Please install it so the report can be generated.")
    cmd = [allure_exe, "generate", str(results_dir), "-o", str(report_dir), "--clean"]
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


def _run_allure_visualizer(results_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str] | None:
    if not VISUALIZER_SCRIPT.exists():
        logger.warning("Allure visualizer script not found at %s; skipping chart generation.", VISUALIZER_SCRIPT)
        return None

    cmd = [
        sys.executable,
        str(VISUALIZER_SCRIPT),
        "--results-dir",
        str(results_dir),
        "--interactive",
        "--interactive-only",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            "Allure visualizer failed with exit code %s: %s",
            result.returncode,
            _shorten_output(result.stderr or result.stdout),
        )
    else:
        charts_dir = results_dir.parent / "allure_charts"
        logger.info("Interactive charts written to %s", charts_dir)
    return result


def _resolve_requested_test_file(tests_dir: Path, test_path: str | None) -> Path:
    default = Path("ui_scripts") / "ui_script_1.py"
    requested = Path(test_path or default)

    if requested.is_absolute():
        raise HTTPException(status_code=400, detail="Absolute test paths are not allowed.")

    if requested.parts and requested.parts[0].lower() == "tests":
        requested = Path(*requested.parts[1:])

    candidate = (tests_dir / requested).resolve()
    tests_dir_resolved = tests_dir.resolve()

    if not str(candidate).startswith(str(tests_dir_resolved)):
        raise HTTPException(
            status_code=403,
            detail="Requested test path must stay within the tests directory.",
        )

    return candidate


def _execute_pytest_script(
    pytest_file: Path,
    src_dir: Path,
    allure_results_dir: Path,
    env: dict[str, str],
) -> None:
    pytest_result = _run_pytest(pytest_file, allure_results_dir, src_dir, env)
    if pytest_result.returncode != 0:
        detail = _shorten_output(pytest_result.stdout or pytest_result.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Pytest run failed for {pytest_file}: {detail or 'see server logs for details'}",
        )


def _discover_test_scripts(tests_dir: Path) -> list[Path]:
    scripts = sorted(tests_dir.glob("**/*.py"))
    filtered = []
    for script in scripts:
        if script.is_file() and not script.name.startswith("__"):
            filtered.append(script.relative_to(tests_dir))
    return filtered


@router.get("/run")
def run_tests(
    test: str | None = Query(
        None,
        description="Relative path (under tests/) of the pytest script to run. Defaults to ui_scripts/ui_script_1.py.",
    ),
    keep_browser: str | None = None,
):
    src_dir = _resolve_src_dir()
    tests_dir = src_dir / "tests"
    if not tests_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tests directory not found.")

    pytest_file = _resolve_requested_test_file(tests_dir, test)
    if not pytest_file.exists():
        raise HTTPException(status_code=404, detail=f"Pytest file not found at {pytest_file}")

    allure_results_dir = src_dir / "allure-results"
    allure_report_dir = src_dir / "allure-report"
    allure_results_dir.mkdir(parents=True, exist_ok=True)
    allure_report_dir.mkdir(parents=True, exist_ok=True)
    project_dir = src_dir.parents[1] if len(src_dir.parents) > 1 else src_dir.parent
    env = _ensure_env(src_dir, project_dir, keep_browser)
    _execute_pytest_script(pytest_file, src_dir, allure_results_dir, env)

    report_result = _generate_report(allure_results_dir, allure_report_dir, src_dir, env)
    if report_result.returncode != 0:
        detail = _shorten_output(report_result.stderr or report_result.stdout)
        raise HTTPException(status_code=500, detail=f"Allure report generation failed: {detail or 'see server logs for details'}")

    _run_allure_visualizer(allure_results_dir, env)

    try:
        summary = collect_run_summary(allure_results_dir)
        if summary:
            history_path = src_dir / "history.json"
            store = MetricsStore(history_path=history_path)
            store.record_run(summary)
    except Exception as exc:
        logger.warning("Failed to update metrics history: %s", exc)

    return {
        "status": "ok",
        "allure_results": str(allure_results_dir),
        "allure_report": str(allure_report_dir),
        "report_url": "/reports/view",
        "allure_charts": str(allure_results_dir.parent / "allure_charts"),
    }

@router.get("/run-all")
def run_all_tests(keep_browser: str | None = None):
    src_dir = _resolve_src_dir()
    tests_dir = src_dir / "tests"
    if not tests_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tests directory not found.")

    discovered = _discover_test_scripts(tests_dir)
    if not discovered:
        raise HTTPException(status_code=404, detail="No pytest scripts found to execute.")

    allure_results_dir = src_dir / "allure-results"
    allure_report_dir = src_dir / "allure-report"
    allure_results_dir.mkdir(parents=True, exist_ok=True)
    allure_report_dir.mkdir(parents=True, exist_ok=True)
    project_dir = src_dir.parents[1] if len(src_dir.parents) > 1 else src_dir.parent
    env = _ensure_env(src_dir, project_dir, keep_browser)

    for rel_path in discovered:
        pytest_file = (tests_dir / rel_path).resolve()
        _execute_pytest_script(pytest_file, src_dir, allure_results_dir, env)

    report_result = _generate_report(allure_results_dir, allure_report_dir, src_dir, env)
    if report_result.returncode != 0:
        detail = _shorten_output(report_result.stderr or report_result.stdout)
        raise HTTPException(status_code=500, detail=f"Allure report generation failed: {detail or 'see server logs for details'}")

    _run_allure_visualizer(allure_results_dir, env)

    try:
        summary = collect_run_summary(allure_results_dir)
        if summary:
            history_path = src_dir / "history.json"
            store = MetricsStore(history_path=history_path)
            store.record_run(summary)
    except Exception as exc:
        logger.warning("Failed to update metrics history: %s", exc)

    return {
        "status": "ok",
        "allure_results": str(allure_results_dir),
        "allure_report": str(allure_report_dir),
        "report_url": "/reports/view",
        "allure_charts": str(allure_results_dir.parent / "allure_charts"),
        "runs_executed": [str(tests_dir / rel) for rel in discovered],
    }


@router.get("/report")
def report():
    return {"report_url": "/reports/view"}


@router.get("/open")
def open_report():
    return {"report_url": "/reports/view"}
