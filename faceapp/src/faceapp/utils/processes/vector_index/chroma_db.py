import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
import ulid
from chromadb import Documents, EmbeddingFunction, Embeddings

from faceapp._base.base import ProcessOutput
from faceapp._base.indexer import Indexer
from faceapp.utils.processes.process_outputs import ChromaLoadOutput


class ChromadbVectorStore(Indexer):
    def __init__(self):
        # TODO: Add logic to connect to a cloud chromadb, if connection
        # details given, else use persistent client
        self.index_client = chromadb.PersistentClient()

    def _create_index(
        self, index_name: str, embedding_model: str = None, index_config: dict = None
    ):
        collections = {i.name for i in self.index_client.list_collections()}
        if index_name not in collections:
            print("Creating index", index_name)
            if not index_config:
                config = {
                    "hnsw": {
                        "space": "cosine",
                        "ef_construction": 500,
                        "ef_search": 500,
                        "max_neighbors": 128,
                    }
                }
            else:
                config = index_config.get(
                    embedding_model,
                    {
                        "hnsw": {
                            "space": "cosine",
                            "ef_construction": 500,
                            "ef_search": 500,
                            "max_neighbors": 128,
                        }
                    },
                )
            print(config)
            collection = self.index_client.create_collection(
                name=index_name,
                configuration=config,
                metadata={
                    "created_on": datetime.now().isoformat(),
                    "updated_on": datetime.now().isoformat(),
                },
            )
        else:
            print("Index exists:", index_name)
            collection = self.index_client.get_collection(name=index_name)
        return collection

    async def load(
        self, embeddings: list, metadata: list, index_config: dict = None, **kwargs
    ) -> ProcessOutput:

        index_names = {d["index_name"] for d in metadata}
        index_embeddings = {
            i: {j: [] for j in ["embeddings", "metadata", "ids"]} for i in index_names
        }
        for i, meta in enumerate(metadata):
            index_name = meta["index_name"]
            index_embeddings[index_name]["embeddings"].append(embeddings[i])
            del meta["index_name"]
            index_embeddings[index_name]["metadata"].append(meta)
            index_embeddings[index_name]["ids"].append(ulid.ulid())

        for index in index_names:
            self.add_data(
                index_name=index,
                ids=index_embeddings[index]["ids"],
                embeddings=index_embeddings[index]["embeddings"],
                metadata=index_embeddings[index]["metadata"],
                index_config=index_config,
            )

        return ChromaLoadOutput(documents=123)

    def add_data(
        self,
        index_name: str,
        ids: list,
        embeddings: list,
        metadata: list,
        index_config: dict = None,
    ):
        assert len(ids) == len(embeddings) == len(metadata)
        embedding_model = metadata[0].get("embedding_model")
        collection = self._create_index(index_name, embedding_model, index_config)
        collection.add(ids=ids, metadatas=metadata, embeddings=embeddings)
        # Update collection metadata
        collection_metadata = collection.metadata
        collection_metadata["updated_on"] = datetime.now().isoformat()
        self.index_client.get_collection(index_name).modify(
            metadata=collection_metadata
        )

    async def search(
        self,
        index_name: str,
        query_embedding: List[float],
        k: Optional[int] = None,
        filters: Optional[str] = None,
        threshold: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified search (cosine).
        - k: top results (default 50 if None).
        - filters: OData filter string, e.g. "metadata eq 'person'".
        - threshold: cosine similarity threshold.
        """
        collection = self.index_client.get_collection(index_name)
        results = collection.query(
            query_embeddings=[query_embedding],
        )
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        print(results)
        formatted = list(filter(lambda i: i[1] < threshold, zip(metadatas, distances)))
        print(formatted)
        return [i[0] for i in formatted]
