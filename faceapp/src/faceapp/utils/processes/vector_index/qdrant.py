from collections import defaultdict
from typing import Any, Dict, List, Optional

import ulid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from faceapp._base.base import ProcessOutput
from faceapp._base.indexer import Indexer
from faceapp.utils.processes.process_outputs import QdrantLoadOutput


class QdrantVectorStore(Indexer):
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
    ):
        if url:
            self.index_client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.index_client = QdrantClient(path=path)
        else:
            self.index_client = QdrantClient(path="./qdrant")

    def _create_index(
        self,
        index_name: str,
        embedding_size: int,
        embedding_model: Optional[str] = None,
        index_config: Optional[dict] = None,
    ):
        if self.index_client.collection_exists(collection_name=index_name):
            return

        resolved_config = self._resolve_config(
            embedding_model=embedding_model,
            index_config=index_config,
        )
        distance_name = resolved_config.get("distance", "cosine").upper()
        distance = getattr(Distance, distance_name, Distance.COSINE)
        hnsw_cfg = resolved_config.get("hnsw", {})
        hnsw_config = HnswConfigDiff(
            m=hnsw_cfg.get("m"),
            ef_construct=hnsw_cfg.get("ef_construct"),
            full_scan_threshold=hnsw_cfg.get("full_scan_threshold"),
        )

        self.index_client.create_collection(
            collection_name=index_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=distance,
            ),
            hnsw_config=hnsw_config,
        )

    async def load(
        self,
        embeddings: list,
        metadata: list,
        index_config: Optional[dict] = None,
        **kwargs,
    ) -> ProcessOutput:
        grouped = defaultdict(lambda: {"embeddings": [], "metadata": [], "ids": []})

        for index, meta in enumerate(metadata):
            payload = dict(meta)
            index_name = payload.pop("index_name")
            grouped[index_name]["embeddings"].append(embeddings[index])
            grouped[index_name]["metadata"].append(payload)
            grouped[index_name]["ids"].append(str(ulid.ulid()))

        documents_by_index: dict[str, int] = {}
        for index_name, data in grouped.items():
            embedding_model = None
            if data["metadata"]:
                embedding_model = data["metadata"][0].get("embedding_model")
            inserted = self.add_data(
                index_name=index_name,
                ids=data["ids"],
                embeddings=data["embeddings"],
                metadata=data["metadata"],
                embedding_model=embedding_model,
                index_config=index_config,
            )
            documents_by_index[index_name] = inserted

        return QdrantLoadOutput(documents=documents_by_index)

    def add_data(
        self,
        index_name: str,
        ids: list,
        embeddings: list,
        metadata: list,
        embedding_model: Optional[str] = None,
        index_config: Optional[dict] = None,
    ) -> int:
        assert len(ids) == len(embeddings) == len(metadata)
        if not embeddings:
            return 0

        self._create_index(
            index_name=index_name,
            embedding_size=len(embeddings[0]),
            embedding_model=embedding_model,
            index_config=index_config,
        )

        points = [
            PointStruct(id=ids[i], vector=embeddings[i], payload=metadata[i])
            for i in range(len(ids))
        ]
        self.index_client.upsert(collection_name=index_name, points=points)
        return len(points)

    async def search(
        self,
        index_name: str,
        query_embedding: List[float] | List[List[float]],
        k: Optional[int] = None,
        filters: Optional[Filter] = None,
        threshold: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        query_embeddings = self._normalize_query_embeddings(query_embedding)
        merged_by_id: dict[str, Dict[str, Any]] = {}
        search_cfg = (config or {}).get("search", {})
        search_params = SearchParams(
            hnsw_ef=search_cfg.get("hnsw_ef"),
            exact=search_cfg.get("exact"),
        )

        for vector in query_embeddings:
            response = self.index_client.query_points(
                collection_name=index_name,
                query=vector,
                query_filter=filters,
                limit=k or 50,
                search_params=search_params,
                with_payload=True,
                with_vectors=False,
                score_threshold=threshold,
            )
            for point in response.points:
                payload = dict(point.payload or {})
                payload["_score"] = point.score
                payload["_id"] = str(point.id)

                existing = merged_by_id.get(payload["_id"])
                if existing is None or payload["_score"] > existing.get("_score", -1.0):
                    merged_by_id[payload["_id"]] = payload

        return list(merged_by_id.values())

    @staticmethod
    def _normalize_query_embeddings(
        query_embedding: List[float] | List[List[float]],
    ) -> List[List[float]]:
        if not query_embedding:
            return []

        first = query_embedding[0]
        if isinstance(first, (int, float)):
            return [query_embedding]  # type: ignore[list-item]

        return query_embedding  # type: ignore[return-value]

    @staticmethod
    def _resolve_config(
        embedding_model: Optional[str],
        index_config: Optional[dict],
    ) -> dict:
        if not index_config:
            return {}

        defaults = index_config.get("defaults", {})
        if not embedding_model:
            return defaults

        model_cfg = index_config.get(embedding_model, {})
        resolved = dict(defaults)
        for key, value in model_cfg.items():
            if isinstance(value, dict) and isinstance(resolved.get(key), dict):
                resolved[key] = {**resolved[key], **value}
            else:
                resolved[key] = value
        return resolved
