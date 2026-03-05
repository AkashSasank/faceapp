from __future__ import annotations

"""FastAPI endpoints that trigger local worker extraction tasks."""

import asyncio
import os
import shutil
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
from faceapp_services.workers.queue import build_task_queue
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="FaceApp Worker API", version="0.1.0")
SHARED_DIR = Path(os.getenv("FACEAPP_SHARED_DIR", "/shared"))
QUEUE_BACKEND = os.getenv("TASK_QUEUE_BACKEND", "redis")
QUEUE_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "faceapp:tasks")
task_queue = build_task_queue(
    backend=QUEUE_BACKEND,
    url=QUEUE_URL,
    queue_name=QUEUE_NAME,
)


def _save_upload(file: UploadFile, folder: Path) -> str:
    """Persist an uploaded file into a folder and return its saved path."""

    file_path = folder / (file.filename or f"upload-{ulid.ulid()}.bin")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_path)


async def _enqueue_task(task: TaskEnvelope) -> WorkerResult:
    """Push a task envelope to the configured queue and return a receipt."""

    await asyncio.to_thread(task_queue.enqueue, task)
    return WorkerResult(
        task_id=task.id,
        task_type=task.type,
        status="success",
        data={
            "queued": True,
            "queue_name": QUEUE_NAME,
        },
    )


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
    temp_dir = SHARED_DIR / f"faceapp-image-{ulid.ulid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        image_path = _save_upload(image, temp_dir)
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
        return await _enqueue_task(task)
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
    temp_dir = SHARED_DIR / f"faceapp-folder-{ulid.ulid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = [_save_upload(file=file, folder=temp_dir) for file in files]

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
        return await _enqueue_task(task)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
