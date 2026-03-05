from __future__ import annotations

"""FastAPI endpoints that trigger local worker extraction tasks."""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import ulid
from faceapp_services.workers.contracts import (
    ExtractionBatchPayload,
    ExtractionUnitPayload,
    TaskEnvelope,
    TaskType,
    WorkerResult,
)
from faceapp_services.workers.dispatcher import WorkerDispatcher
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="FaceApp Worker API", version="0.1.0")
dispatcher = WorkerDispatcher()


def _save_upload(file: UploadFile, folder: Path) -> str:
    file_path = folder / (file.filename or f"upload-{ulid.ulid()}.bin")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_path)


@app.get("/health")
def health() -> dict:
    """Simple health endpoint for local service checks."""
    return {"status": "ok"}


@app.post("/extract/image", response_model=WorkerResult)
async def extract_image(
    image: UploadFile = File(...),
    embedding_models: List[str] = Form(...),
    features: List[str] = Form(...),
    face_detector: str = Form("mtcnn"),
    project_id: Optional[str] = Form(None),
) -> WorkerResult:
    """Upload one image and run the unit extraction worker task."""
    temp_dir = tempfile.mkdtemp(prefix="faceapp-image-")
    try:
        image_path = _save_upload(image, Path(temp_dir))
        payload = ExtractionUnitPayload(
            path=image_path,
            embedding_models=embedding_models,
            features=features,
            face_detector=face_detector,
            project_id=project_id,
        )
        task = TaskEnvelope(
            id=str(ulid.ulid()),
            type=TaskType.EXTRACT_UNIT,
            payload=payload,
            source="fastapi",
        )
        return await dispatcher.dispatch(task)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/extract/folder", response_model=WorkerResult)
async def extract_folder(
    files: List[UploadFile] = File(...),
    embedding_models: List[str] = Form(...),
    features: List[str] = Form(...),
    face_detector: str = Form("mtcnn"),
    project_id: Optional[str] = Form(None),
) -> WorkerResult:
    """Upload many images and run the batch extraction worker task."""
    temp_dir = tempfile.mkdtemp(prefix="faceapp-folder-")
    try:
        temp_path = Path(temp_dir)
        paths = [_save_upload(file=file, folder=temp_path) for file in files]

        payload = ExtractionBatchPayload(
            paths=paths,
            embedding_models=embedding_models,
            features=features,
            face_detector=face_detector,
            project_id=project_id,
        )
        task = TaskEnvelope(
            id=str(ulid.ulid()),
            type=TaskType.EXTRACT_BATCH,
            payload=payload,
            source="fastapi",
        )
        return await dispatcher.dispatch(task)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
