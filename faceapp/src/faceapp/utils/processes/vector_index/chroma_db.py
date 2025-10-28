import os
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

import chromadb
import ulid
from chromadb import EmbeddingFunction, Documents, Embeddings
from faceapp._base.indexer import Indexer


class ChromadbVectorStore(Indexer):
    def __init__(
            self
    ):
        self.index_client = chromadb.Client()
        self.collections = {}

    def _create_index(self, index_name: str):
        print("Creating collection: ", index_name)
        collection = self.index_client.get_or_create_collection(name=index_name,
                                                                configuration={
                                                                    "hnsw":{
                                                                        "space":"cosine",
                                                                        "ef_construction": 300,
                                                                        "ef_search": 300
                                                                    }
                                                                },
                                                                metadata={
                                                                    "created_on": datetime.now().isoformat(),
                                                                    "updated_on": datetime.now().isoformat(),
                                                                })
        self.collections[index_name] = collection
        return collection


    async def load(self, embeddings: list, metadata:list, **kwargs) -> dict:

        index_names = {d['index_name'] for d in metadata}
        index_embeddings = {i:{j:[] for j in ["embeddings", "metadata", "ids"]} for i in index_names}
        for i, meta in enumerate(metadata):
            print(meta)
            index_name = meta["index_name"]
            index_embeddings[index_name]["embeddings"].append(embeddings[i])
            del meta["index_name"]
            index_embeddings[index_name]["metadata"].append(meta)
            index_embeddings[index_name]["ids"].append(ulid.ulid())

        for index in index_names:
            collection = self._create_index(index)
            collection.add(ids=index_embeddings[index]['ids'],
                           embeddings=index_embeddings[index]['embeddings'],
                           metadatas=index_embeddings[index]['metadata'])

        return {"documents": 123}


    async def search(
            self,
            index_name: str,
            query_embedding: List[float],
            k: Optional[int] = None,
            filters: Optional[str] = None,
            threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified search (cosine).
        - k: top results (default 50 if None).
        - filters: OData filter string, e.g. "metadata eq 'person'".
        - threshold: cosine similarity threshold.
        """
        fetch_k = k or 50
        search_client = self.create_search_client(index_name)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=fetch_k,  # Example: retrieve 5 nearest neighbors
            fields="embedding",
        )
        results = search_client.search(
            vector_queries=[vector_query],
            select=["blob_name", "id"],
            include_total_count=True,
        )
        formatted = list(filter(lambda i: i["@search.score"] > threshold, results))
        return formatted
