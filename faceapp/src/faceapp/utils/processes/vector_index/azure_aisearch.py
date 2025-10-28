import asyncio
import enum
import os
from typing import Any, Dict, List, Optional

import ulid
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, ExhaustiveKnnAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from faceapp._base.indexer import Indexer


class AzureAISearchVectorStore(Indexer):
    META_ID = "__meta__"

    def __init__(
        self,
        service_name: str = os.environ.get("AZURE_AI_SEARCH_SERVICE_NAME"),
        api_key: str = os.environ.get("AZURE_AI_SEARCH_API_KEY"),
    ):
        self.endpoint = f"https://{service_name}.search.windows.net"
        self.credential = AzureKeyCredential(api_key)

        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

        self.search_clients = {}

    def _create_index(self, index_name: str, dim: int):
        if index_name in [i.name for i in self.index_client.list_indexes()]:
            return

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-1",
                ),
                ExhaustiveKnnAlgorithmConfiguration(name="xknn")
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile-hnsw-scalar",
                    algorithm_configuration_name="hnsw-1",
                ),
                VectorSearchProfile(
                    name="vector-profile-xknn",
                    algorithm_configuration_name="xknn",
                )
            ],
        )
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dim,  # Replace with your embedding model's dimensions
                vector_search_profile_name="vector-profile-xknn",
            ),
            SimpleField(name="blob_name", type=SearchFieldDataType.String),
        ]

        # Create the SearchIndex object
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self.index_client.create_index(index)
        print("Creating index: ", index_name)
        self.create_search_client(index_name)

    def create_search_client(self, index_name):
        search_client = SearchClient(
            endpoint=self.endpoint, index_name=index_name, credential=self.credential
        )
        self.search_clients[index_name] = search_client
        return search_client

    async def load(self, extractions: list, *args, **kwargs) -> dict:
        tasks = [self.__insert(**x) for x in extractions]
        docs = await asyncio.gather(*tasks)
        return {"documents": docs, "blob_name": extractions[0].get("blob_name")}

    async def __insert(
        self,
        index_name: str,
        embedding: List[float],
        doc_id: Optional[str] = None,
        blob_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs
    ) -> dict:
        doc_id = doc_id or str(ulid.ulid())
        # now = datetime.datetime.now().isoformat()

        doc = {
            "id": doc_id,
            "embedding": embedding,
            "blob_name": blob_name,
        }
        # Create index and search client
        self._create_index(index_name, len(embedding))
        if not self.search_clients.get(index_name):
            self.create_search_client(index_name)

        self.search_clients[index_name].upload_documents(documents=[doc])
        return {"document_id": doc_id, "index": index_name}

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
