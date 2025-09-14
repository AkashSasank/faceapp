import asyncio
import datetime
from typing import Any, Dict, List, Optional

import ulid
from elasticsearch import Elasticsearch

from faceapp._base.indexer import Indexer


class ElasticVectorStore(Indexer):
    META_ID = "__meta__"

    def __init__(self, host: str = "http://localhost:9200"):
        self.es = Elasticsearch(host)

    def _create_index(self, dim: int, index_name: str):
        index_name = index_name.lower()
        if self.es.indices.exists(index=index_name):
            return

        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                }
            },
        )

        # Create metadata doc
        now = datetime.datetime.now().isoformat()
        self.es.index(
            index=index_name,
            id=self.META_ID,
            document={"created_at": now, "updated_at": now, "doc_count": 0},
        )
        self.es.indices.refresh(index=index_name)

    def _update_index_metadata(self, index_name):
        now = datetime.datetime.now().isoformat()
        script = {
            "source": "ctx._source.updated_at = params.time; ctx._source.doc_count += 1",
            "params": {"time": now},
        }
        self.es.update(index=index_name, id=self.META_ID, script=script)

    async def load(self, extractions: list, *args, **kwargs) -> dict:

        tasks = [self.__insert(**extraction) for extraction in extractions]

        docs = await asyncio.gather(*tasks)
        return {"documents": docs, "input_path": extractions[0].get("content")}

    async def __insert(
        self,
        index_name: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> dict:
        dim = len(embedding)
        self._create_index(dim=dim, index_name=index_name)
        doc_id = doc_id or str(ulid.ulid())
        now = datetime.datetime.now().isoformat()

        body = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        self.es.index(index=index_name, id=doc_id, document=body)
        self.es.indices.refresh(index=index_name)
        self._update_index_metadata(index_name=index_name)
        return {
            "document_id": doc_id,
            "index": index_name,
        }

    def search_ann(
        self,
        index_name: str,
        query_embedding: List[float],
        k: int = 5,
        num_candidates: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN vector search (fast, approximate).
        """
        query_body: Dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": num_candidates,
            },
            "_source": ["content", "metadata", "created_at", "updated_at"],
        }

        if filters:
            query_body["query"] = {
                "bool": {
                    "filter": [
                        {"term": {f"metadata.{k}": v}} for k, v in filters.items()
                    ]
                }
            }

        res = self.es.search(index=index_name, **query_body)
        return self._format_results(res)

    def search_cosine(
        self,
        index_name: str,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Exact cosine similarity search using script_score.
        Slower than ANN but precise.
        """
        query: Dict[str, Any] = {"match_all": {}}
        if filters:
            query = {
                "bool": {
                    "filter": [
                        {"term": {f"metadata.{k}": v}} for k, v in filters.items()
                    ]
                }
            }

        res = self.es.search(
            index=index_name,
            size=k,
            query={
                "script_score": {
                    "query": query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding},
                    },
                }
            },
            _source=["content", "metadata", "created_at", "updated_at"],
        )
        return self._format_results(res)

    @staticmethod
    def _format_results(res):
        results = []
        for hit in res["hits"]["hits"]:
            results.append(
                {
                    "id": hit["_id"],
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "created_at": hit["_source"].get("created_at"),
                    "updated_at": hit["_source"].get("updated_at"),
                    "score": hit["_score"],
                }
            )
        return results

    def get_index_metadata(self, index_name) -> Dict[str, Any]:
        doc = self.es.get(index=index_name, id=self.META_ID, ignore=[404])
        return doc["_source"] if doc.get("found") else {}
