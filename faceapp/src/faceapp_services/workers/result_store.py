from __future__ import annotations

"""Qdrant-backed storage for worker execution results."""

from datetime import datetime
from typing import Optional

import ulid
from faceapp_services.workers.contracts import WorkerResult
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantWorkerResultStore:
    """Stores worker task results as payload documents in Qdrant."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: str = "faceapp_worker_results",
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
        """Create the backing collection when it does not yet exist."""

        if self.client.collection_exists(collection_name=self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def write_result(self, result: WorkerResult, worker_id: str) -> str:
        """Persist one worker result as a payload document in Qdrant."""

        point_id = str(ulid.ulid())
        payload = result.model_dump()
        payload["worker_id"] = worker_id
        payload["created_on"] = datetime.now().isoformat()

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
