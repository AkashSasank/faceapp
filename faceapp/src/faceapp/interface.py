from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from typing import Any, Optional

import ulid
from faceapp_services.vector_db import apply_runtime_connection
from faceapp_services.vector_db import (
    models_requiring_embedding as resolved_models_requiring_embedding,
)
from faceapp_services.workers.contracts import (
    ExtractionUnitPayload,
    TaskEnvelope,
    TaskType,
)
from faceapp_services.workers.dispatcher import WorkerDispatcher

from faceapp.utils.processes.vector_index.qdrant import QdrantVectorStore
from faceapp.utils.search import FaceSearch

_RUNTIME_ENV_KEYS = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_PATH",
)


def _clean_optional_str(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    return cleaned or None


@contextmanager
def _runtime_connection_scope(
    *,
    config_name: str | None,
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_path: Optional[str],
):
    """Temporarily align SDK runtime connection settings with manager flow."""

    previous_env = {key: os.environ.get(key) for key in _RUNTIME_ENV_KEYS}
    cleaned_url = _clean_optional_str(qdrant_url)
    cleaned_api_key = _clean_optional_str(qdrant_api_key)
    cleaned_path = _clean_optional_str(qdrant_path)

    try:
        if cleaned_url or cleaned_api_key or cleaned_path:
            if cleaned_url:
                os.environ["QDRANT_URL"] = cleaned_url
                os.environ.pop("QDRANT_PATH", None)
            elif cleaned_path:
                os.environ["QDRANT_PATH"] = cleaned_path
                os.environ.pop("QDRANT_URL", None)
            else:
                os.environ.pop("QDRANT_URL", None)
                os.environ.pop("QDRANT_PATH", None)

            if cleaned_api_key:
                os.environ["QDRANT_API_KEY"] = cleaned_api_key
            else:
                os.environ.pop("QDRANT_API_KEY", None)
        else:
            apply_runtime_connection(config_name)

        yield
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FaceAppClient:
    """Programmatic ingest/search interface with explicit parameters."""

    def ingest(
        self,
        *,
        file_path: str,
        project_id: str,
        embedding_models: list[str],
        features: list[str],
        face_detector: str,
        source: str = "sdk",
        metadata: Optional[dict[str, Any]] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if not file_path or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not project_id or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(embedding_models, list) or not embedding_models:
            raise ValueError("embedding_models must be a non-empty list[str]")
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a non-empty list[str]")
        if not isinstance(face_detector, str) or not face_detector.strip():
            raise ValueError("face_detector must be a non-empty string")

        file_path = file_path.strip()
        project_id = project_id.strip()
        embedding_models = [str(model) for model in embedding_models]
        features = [str(feature) for feature in features]
        face_detector = face_detector.strip()
        if isinstance(source, str) and source.strip():
            source = source.strip()
        else:
            source = "sdk"

        indexed_meta: dict[str, Any] = {}
        if isinstance(metadata, dict):
            indexed_meta.update(metadata)

        with _runtime_connection_scope(
            config_name=project_id,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_path=qdrant_path,
        ):
            pending_models = resolved_models_requiring_embedding(
                file_path=file_path,
                project_id=project_id,
                embedding_models=embedding_models,
                config_name=project_id,
            )
            skipped_models = [
                model for model in embedding_models if model not in pending_models
            ]

            if not pending_models:
                return {
                    "status": "skipped",
                    "reason": "already_indexed",
                    "file_path": file_path,
                    "project_id": project_id,
                    "skipped_models": skipped_models,
                }

            payload = ExtractionUnitPayload(
                path=file_path,
                embedding_models=pending_models,
                features=features,
                face_detector=face_detector,
                project_id=project_id,
                meta=indexed_meta,
            )
            task = TaskEnvelope(
                id=str(ulid.ulid()),
                type=TaskType.EXTRACT_UNIT,
                payload=payload,
                source=source,
            )
            dispatch_result = asyncio.run(WorkerDispatcher().dispatch(task))
            worker_result = dispatch_result.model_dump()
            return {
                "status": "processed",
                "file_path": file_path,
                "project_id": project_id,
                "requested_models": embedding_models,
                "processed_models": pending_models,
                "skipped_models": skipped_models,
                "metadata": indexed_meta,
                "result": worker_result,
            }

    def search(
        self,
        *,
        file_path: str,
        project_id: str,
        embedding_models: list[str],
        thresholds: list[float],
        top_k: Optional[int] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if not file_path or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not project_id or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(embedding_models, list) or not embedding_models:
            raise ValueError("embedding_models must be a non-empty list[str]")
        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError("thresholds must be a non-empty list[float]")

        file_path = file_path.strip()
        project_id = project_id.strip()
        embedding_models = [str(model) for model in embedding_models]
        thresholds = [float(value) for value in thresholds]

        if len(thresholds) == 1 and len(embedding_models) > 1:
            thresholds = thresholds * len(embedding_models)
        if len(thresholds) != len(embedding_models):
            raise ValueError("thresholds length must match embedding_models")

        vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            path=qdrant_path,
        )
        finder = FaceSearch(
            vector_db=vector_store,
            embedding_models=embedding_models,
            model_thresholds=thresholds,
            top_k=top_k,
            project_id=project_id,
        )
        matches = asyncio.run(finder.find(file_path))
        return {
            "query": file_path,
            "project_id": project_id,
            "embedding_models": embedding_models,
            "thresholds": thresholds,
            "top_k": top_k,
            "matches": matches,
        }


def ingest_file(
    *,
    file_path: str,
    project_id: str,
    embedding_models: list[str],
    features: list[str],
    face_detector: str,
    source: str = "sdk",
    metadata: Optional[dict[str, Any]] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_path: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience function for one-file ingestion with explicit params."""

    return FaceAppClient().ingest(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        features=features,
        face_detector=face_detector,
        source=source,
        metadata=metadata,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_path=qdrant_path,
    )


def search_file(
    *,
    file_path: str,
    project_id: str,
    embedding_models: list[str],
    thresholds: list[float],
    top_k: Optional[int] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_path: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience function for one-file search with explicit params."""

    return FaceAppClient().search(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        thresholds=thresholds,
        top_k=top_k,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_path=qdrant_path,
    )
