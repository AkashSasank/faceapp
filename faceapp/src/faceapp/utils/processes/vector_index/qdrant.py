import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from faceapp._base.base import ProcessOutput
from faceapp._base.indexer import Indexer
from faceapp.utils.processes.process_outputs import QdrantLoadOutput


def build_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    path: Optional[str] = None,
) -> QdrantClient:
    resolved_url = url.strip() if isinstance(url, str) else None
    resolved_path = path.strip() if isinstance(path, str) else None

    if resolved_url:
        return QdrantClient(url=resolved_url, api_key=api_key)
    if resolved_path:
        return QdrantClient(path=resolved_path)
    return QdrantClient(path="./qdrant")


def collection_name_for_model(project_id: Optional[str], model: str) -> str:
    normalized_model = model.lower()
    if project_id:
        normalized_project = project_id.strip().replace(" ", "_")
        return f"{normalized_project}_{normalized_model}"
    return normalized_model


def compute_file_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def should_skip_file_before_embedding(
    file_path: str,
    project_id: Optional[str],
    embedding_models: list[str],
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    path: Optional[str] = None,
) -> bool:
    models_to_process = models_requiring_embedding(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        url=url,
        api_key=api_key,
        path=path,
    )
    return len(models_to_process) == 0


def models_requiring_embedding(
    file_path: str,
    project_id: Optional[str],
    embedding_models: list[str],
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    path: Optional[str] = None,
) -> list[str]:
    if not embedding_models:
        return []

    target = Path(file_path)
    if not target.is_file():
        return list(embedding_models)

    try:
        image_hash = compute_file_sha256(file_path)
        client = build_qdrant_client(url=url, api_key=api_key, path=path)
        pending_models: list[str] = []

        for model in embedding_models:
            collection_name = collection_name_for_model(project_id, model)
            if not client.collection_exists(collection_name=collection_name):
                pending_models.append(model)
                continue

            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="image_hash",
                            match=MatchValue(value=image_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                pending_models.append(model)
        return pending_models
    except Exception:
        return list(embedding_models)


class QdrantVectorStore(Indexer):
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.index_client = build_qdrant_client(
            url=url,
            api_key=api_key,
            path=path,
        )

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

        try:
            self.index_client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(
                    size=embedding_size,
                    distance=distance,
                ),
                hnsw_config=hnsw_config,
            )
        except Exception:
            if self.index_client.collection_exists(collection_name=index_name):
                return
            raise

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
            grouped[index_name]["ids"].append(str(uuid4()))

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

    def existing_dedupe_hashes(
        self,
        index_name: str,
        dedupe_hashes: set[str],
    ) -> set[str]:
        if not dedupe_hashes:
            return set()

        if not self.index_client.collection_exists(collection_name=index_name):
            return set()

        existing_hashes: set[str] = set()
        for dedupe_hash in dedupe_hashes:
            points, _ = self.index_client.scroll(
                collection_name=index_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="dedupe_hash",
                            match=MatchValue(value=dedupe_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if points:
                existing_hashes.add(dedupe_hash)

        return existing_hashes

    async def search(
        self,
        index_name: str,
        query_embedding: Union[List[float], List[List[float]]],
        k: Optional[int] = None,
        filters: Optional[Filter] = None,
        threshold: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        if not self.index_client.collection_exists(collection_name=index_name):
            return []

        resolved_limit = self._resolve_query_limit(index_name=index_name, k=k)

        query_embeddings = self._normalize_query_embeddings(query_embedding)
        merged_by_id: dict[str, Dict[str, Any]] = {}
        search_cfg = (config or {}).get("search", {})
        search_params = SearchParams(
            hnsw_ef=search_cfg.get("hnsw_ef"),
            exact=search_cfg.get("exact"),
        )

        for vector in query_embeddings:
            try:
                response = self.index_client.query_points(
                    collection_name=index_name,
                    query=vector,
                    query_filter=filters,
                    limit=resolved_limit,
                    search_params=search_params,
                    with_payload=True,
                    with_vectors=True,
                    score_threshold=threshold,
                )
            except UnexpectedResponse as error:
                if "doesn't exist" in str(error):
                    return []
                raise
            for point in response.points:
                payload = dict(point.payload or {})
                payload["_score"] = point.score
                payload["_id"] = str(point.id)

                existing = merged_by_id.get(payload["_id"])
                if existing is None or payload["_score"] > existing.get(
                    "_score",
                    -1.0,
                ):
                    merged_by_id[payload["_id"]] = payload

        return list(merged_by_id.values())

    def _resolve_query_limit(
        self,
        index_name: str,
        k: Optional[int],
    ) -> int:
        if isinstance(k, int) and k > 0:
            return k

        try:
            collection = self.index_client.get_collection(index_name)
            points_count = collection.points_count
            if isinstance(points_count, int) and points_count > 0:
                return points_count

            vectors_count = getattr(collection, "vectors_count", None)
            if isinstance(vectors_count, int) and vectors_count > 0:
                return vectors_count
        except Exception:
            pass

        return 1_000_000

    @staticmethod
    def _normalize_query_embeddings(
        query_embedding: Union[List[float], List[List[float]]],
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
