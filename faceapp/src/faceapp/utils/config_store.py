from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, UpdateStatus, VectorParams


class QdrantConfigStore:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: str = "faceapp_config_store",
        vector_size: int = 4,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(path="./qdrant")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(project_name: str, namespace: str) -> str:
        return f"{namespace}:{project_name}"

    def upsert_project_config(
        self,
        project_name: str,
        config: dict[str, Any],
        namespace: str = "project_overrides",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        point_id = self._point_id(
            project_name=project_name,
            namespace=namespace,
        )
        payload = {
            "project_name": project_name,
            "namespace": namespace,
            "config": config,
            "updated_on": datetime.now().isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata

        point = PointStruct(
            id=point_id,
            vector=[0.0] * self.vector_size,
            payload=payload,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return point_id

    def get_project_config(
        self,
        project_name: str,
        namespace: str = "project_overrides",
    ) -> Optional[dict[str, Any]]:
        point_id = self._point_id(
            project_name=project_name,
            namespace=namespace,
        )
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None

        payload = points[0].payload or {}
        config = payload.get("config")
        return config if isinstance(config, dict) else None

    def delete_project_config(
        self,
        project_name: str,
        namespace: str = "project_overrides",
    ) -> bool:
        point_id = self._point_id(
            project_name=project_name,
            namespace=namespace,
        )
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        return result.status == UpdateStatus.COMPLETED

    def list_project_configs(
        self,
        namespace: str = "project_overrides",
        limit: int = 100,
    ) -> list[str]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )

        projects: list[str] = []
        for point in points:
            payload = point.payload or {}
            if payload.get("namespace") != namespace:
                continue
            project_name = payload.get("project_name")
            if isinstance(project_name, str):
                projects.append(project_name)
        return projects


def merge_config(
    base_config: dict[str, Any],
    override_config: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged
