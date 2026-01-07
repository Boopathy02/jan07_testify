# image_text_api.py

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import os
import zipfile
import tempfile
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from logic.image_text_extractor import process_image_gpt
from services.graph_service import build_dependency_graph
from utils.match_utils import normalize_page_name
from config.settings import get_data_path
from utils.chroma_client import get_collection
from database.session import get_db
from database.models import Project, ImageMetadata, ImageUploadRun
from database.project_storage import DatabaseBackedProjectStorage
from datetime import datetime
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import re

load_dotenv()

router = APIRouter()

# Logging (project-scoped)
log_path = os.path.join(get_data_path(), "upload_image_logs.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# ChromaDB setup
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _chroma_collection():
    return get_collection("element_metadata", embedding_function=embedding_function)


def _get_active_project(db: Session) -> Project:
    """Resolve the currently active project using env hints."""
    project_id_value = os.environ.get("SMARTAI_PROJECT_ID")
    if project_id_value:
        try:
            project = (
                db.query(Project)
                .filter(Project.id == int(project_id_value))
                .first()
            )
            if project:
                return project
        except ValueError:
            pass

    project_dir = os.environ.get("SMARTAI_PROJECT_DIR")
    if project_dir:
        segment = Path(project_dir).name
        match = re.match(r"(?P<id>\d+)-", segment)
        if match:
            candidate_id = int(match.group("id"))
            project = (
                db.query(Project)
                .filter(Project.id == candidate_id)
                .first()
            )
            if project:
                os.environ["SMARTAI_PROJECT_ID"] = str(project.id)
                return project

        normalized_slug = Project.normalized_key(segment.replace("-", " ").replace("_", " "))
        project = (
            db.query(Project)
            .filter(Project.project_key == normalized_slug)
            .order_by(Project.created_at.desc())
            .first()
        )
        if project:
            os.environ["SMARTAI_PROJECT_ID"] = str(project.id)
            return project

    raise HTTPException(
        status_code=400,
        detail="Active project not found in database. Activate a project before uploading images.",
    )


def _serialize_metadata_list(metadata_list: List[dict]) -> List[dict]:
    """Convert numpy/complex objects to JSON-friendly structures."""

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    return json.loads(json.dumps(metadata_list, default=_default))


def _serialize_results(results: List[dict]) -> List[dict]:
    return json.loads(json.dumps(results))


def _merge_metadata_entries(existing: List[dict], incoming: List[dict]) -> List[dict]:
    """Merge metadata lists, overwriting matches by stable identifiers."""
    merged: List[dict] = []
    index_map = {}

    def _key(item: dict) -> tuple:
        identifier = (
            (item or {}).get("id")
            or (item or {}).get("ocr_id")
            or (item or {}).get("unique_name")
            or (item or {}).get("label_text")
            or ""
        )
        intent = (item or {}).get("intent") or ""
        return (identifier.strip().lower(), intent.strip().lower())

    for entry in existing or []:
        key = _key(entry)
        index_map[key] = len(merged)
        merged.append(entry)

    for entry in incoming or []:
        key = _key(entry)
        if key in index_map:
            merged[index_map[key]] = entry
        else:
            index_map[key] = len(merged)
            merged.append(entry)

    return merged


def _project_root_storage(project: Project, db: Session) -> Optional[DatabaseBackedProjectStorage]:
    project_dir = os.environ.get("SMARTAI_PROJECT_DIR")
    if not project_dir:
        return None
    try:
        root = Path(project_dir).resolve()
    except Exception:
        return None
    return DatabaseBackedProjectStorage(project, root, db)


def _persist_data_file(
    storage: Optional[DatabaseBackedProjectStorage],
    absolute_path: Path,
    payload: str,
    encoding: str = "utf-8",
) -> None:
    if not storage:
        return
    try:
        relative = absolute_path.resolve().relative_to(storage.base_dir.resolve())
    except Exception:
        return
    storage.write_file(relative.as_posix(), payload, encoding or "utf-8")


@router.post("/upload-image")
async def upload_image(
    images: List[UploadFile] = File(...),
    ordered_images: str = Form(None),
    db: Session = Depends(get_db),
):
    # Require an active project so we don't write to repo-level defaults
    if not os.environ.get("SMARTAI_PROJECT_DIR") or not os.environ.get("SMARTAI_SRC_DIR"):
        raise HTTPException(status_code=400, detail="No active project. Start a project first (POST /projects/save-details).")
    project = _get_active_project(db)
    data_storage = _project_root_storage(project, db)
    dp = get_data_path()
    os.makedirs(os.path.join(dp, "regions"), exist_ok=True)
    os.makedirs(os.path.join(dp, "images"), exist_ok=True)
    results = []
    ordered_image_list = []

    # Step 1: Parse frontend ordering
    if ordered_images:
        try:
            parsed_json = json.loads(ordered_images)
            ordered_image_list = parsed_json.get("ordered_images", [])
            ordered_image_list = [os.path.basename(
                f) for f in ordered_image_list]
            logger.info(
                f"🟢 Ordered images from frontend: {ordered_image_list}")
        except Exception as parse_err:
            logger.warning(f"⚠️ Failed to parse ordered_images: {parse_err}")
            ordered_image_list = []

    # Step 2: Extract uploaded files
    temp_dir = tempfile.mkdtemp()
    image_file_map = {}
    actual_received_images = []

    try:
        for file in images:
            filename = file.filename.lower()
            if filename.endswith(".zip"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(await file.read())
                    tmp_zip_path = tmp_zip.name
                with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            else:
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, "wb") as out_file:
                        out_file.write(await file.read())

        # Step 3: Final image order
        extracted_images = [f for f in os.listdir(temp_dir) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        image_names = ordered_image_list if ordered_image_list else sorted(
            extracted_images)

        # Group images by normalized page_name
        page_images = {}
        for image_name in image_names:
            page_name = normalize_page_name(image_name)
            page_images.setdefault(page_name, []).append(image_name)

        # For saving all raw metadata
        # all_raw_metadata = []

        # Step 4: Process images grouped by logical page
        for page_name, image_group in page_images.items():
            # Fetch existing label_texts for this page from chroma
            try:
                existing = _chroma_collection().get(
                    where={"page_name": page_name})
                existing_label_texts = set(
                    m["label_text"].strip().lower()
                    for m in (existing["metadatas"] or [])
                    if m and m.get("label_text")
                )
            except Exception as fetch_err:
                logger.warning(
                    f"⚠️ Failed to fetch existing metadatas for {page_name}: {fetch_err}")
                existing_label_texts = set()

            # For each image for this logical page
            for image_name in image_group:
                image_path = os.path.join(temp_dir, image_name)
                if not os.path.exists(image_path):
                    logger.warning(f"⚠️ Skipping missing image: {image_name}")
                    continue

                with Image.open(image_path) as img:
                    logger.debug(f"📷 Processing image: {image_name}")

                    permanent_image_path = os.path.join(
                        get_data_path(), "images", image_name)
                    img.save(permanent_image_path)

                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # DEBUG_LOG_PATH = f"./data/metadata_logs_{timestamp}.json"

                    # GPT image extraction
                    metadata_list = await process_image_gpt(
                        img, image_name,
                        image_path=permanent_image_path,
                        # debug_log_path=DEBUG_LOG_PATH
                    )
                    metadata_list = metadata_list or []
                    serialized_metadata = _serialize_metadata_list(metadata_list)

                    # Save per-image metadata to data/stored/timestamp_imageName.json
                    base_image_name = os.path.splitext(os.path.basename(image_name))[0]
                    store_dir = Path(get_data_path()) / "stored"
                    store_dir.mkdir(parents=True, exist_ok=True)
                    out_path = store_dir / f"{base_image_name}.json"
                    if out_path.exists():
                        try:
                            existing_meta = json.loads(out_path.read_text(encoding="utf-8") or "[]")
                        except json.JSONDecodeError:
                            existing_meta = []
                        combined = _merge_metadata_entries(existing_meta, serialized_metadata)
                    else:
                        combined = serialized_metadata
                    stored_payload = json.dumps(combined, indent=4, ensure_ascii=False)
                    out_path.write_text(stored_payload, encoding="utf-8")
                    _persist_data_file(data_storage, out_path, stored_payload)

                    record = (
                        db.query(ImageMetadata)
                        .filter(
                            ImageMetadata.project_id == project.id,
                            ImageMetadata.image_name == image_name,
                        )
                        .first()
                    )
                    if record:
                        record.page_name = page_name
                        record.metadata_json = serialized_metadata
                    else:
                        record = ImageMetadata(
                            project_id=project.id,
                            page_name=page_name,
                            image_name=image_name,
                            metadata_json=serialized_metadata,
                        )
                        db.add(record)
                        db.flush()

                    results.append(
                        {
                            "image_name": image_name,
                            "page_name": page_name,
                            "metadata_id": record.id,
                            "metadata_count": len(serialized_metadata),
                        }
                    )

                    # all_raw_metadata.append({
                    #     "image_name": image_name,
                    #     "metadata": metadata_list
                    # })

                    # # Only add new label_texts for this logical page
                    # for metadata in metadata_list:
                    #     original_label_text = metadata.get("label_text", "")
                    #     cleaned_label_text = clean_label_text(
                    #         original_label_text)
                    #     # Overwrite with cleaned version
                    #     metadata["label_text"] = cleaned_label_text
                    #     if cleaned_label_text and cleaned_label_text not in existing_label_texts:
                    #         chroma_collection.add(
                    #             ids=[metadata["id"]],
                    #             documents=[metadata["text"]],
                    #             metadatas=[metadata]
                    #         )
                    #         results.append(metadata)
                    #         existing_label_texts.add(cleaned_label_text)

                image_file_map[image_name] = (image_path, page_name)
                actual_received_images.append(image_name)

        # # Save all raw GPT metadata to a single file
        # raw_data_file_path = os.path.join("data", "raw_data_from_gpt.json")
        # with open(raw_data_file_path, "w", encoding="utf-8") as f:
        #     json.dump(all_raw_metadata, f, indent=2, ensure_ascii=False)
        # logger.info(f"📝 Saved all raw GPT metadata to {raw_data_file_path}")

        # Step 5: Store dependency graph
        if ordered_image_list:
            build_dependency_graph(
                ordered_image_list, output_path=os.path.join(get_data_path(), "dependency_graph.json"))
            logger.info(
                "📄 Dependency graph stored in data/dependency_graph.json")

        # Step 6: Log order metadata
        order_json_path = os.path.join(get_data_path(), "image_order.json")
        with open(order_json_path, "w") as f:
            json.dump({
                "ordered_from_frontend": ordered_image_list,
                "processed_order": actual_received_images
            }, f, indent=2)
        logger.info("[FILES] Ordered images logged to data/image_order.json")

        run_record = ImageUploadRun(
            project_id=project.id,
            results=_serialize_results(results),
            image_count=len(results),
        )
        db.add(run_record)
        db.flush()

        return JSONResponse(content={"status": "success", "data": results, "run_id": run_record.id})

    except Exception as e:
        logger.error("❌ Error in upload_image", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



def clean_label_text(text: str) -> str:
    # Remove leading/trailing numbers, dots, dashes, and spaces
    cleaned = re.sub(r"^[\s\W\d_]+|[\s\W\d_]+$", "", text, flags=re.UNICODE)
    return cleaned
