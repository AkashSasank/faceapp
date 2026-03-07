import asyncio
import itertools
from typing import Any, Dict, List, Optional

from faceapp._base.indexer import Indexer
from faceapp.utils.processes.extractor import FaceEmbedder


class Search:
    def __init__(
        self,
        vector_db: Indexer,
        embedding_model: str,
        threshold: float = 0.6,
        top_k: Optional[int] = None,
    ):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.top_k = top_k
        self.face_embedder = FaceEmbedder()

    async def get_matches(self, img_path: str, index_name: str):
        embeddings = self.face_embedder.represent_faces(
            img_path,
            embedding_model=self.embedding_model,
            face_detector="mtcnn",
        )["embedding_objs"]
        search_tasks = [
            self.vector_db.search(
                index_name=index_name,
                query_embedding=obj["embedding"],
                threshold=self.threshold,
                k=self.top_k,
            )
            for obj in embeddings
        ]
        search_results = await asyncio.gather(*search_tasks)
        normalized_results = [
            self.__normalize_search_results(result) for result in search_results
        ]
        return self.__build_retrievals(
            search_results=normalized_results,
            index_name=index_name,
        )

    @staticmethod
    def __normalize_search_results(
        search_result: Any,
    ) -> List[Dict[str, Any]]:
        if isinstance(search_result, list):
            return [item for item in search_result if isinstance(item, dict)]
        if isinstance(search_result, dict):
            return [search_result]
        return []

    @staticmethod
    def __get_match_result(
        search_result: Dict[str, Any],
        embedding_model: str,
        index_name: str,
        query_face_index: int,
        rank: int,
    ) -> Optional[Dict[str, Any]]:
        img_path = search_result.get("blob_name")
        if not img_path:
            return None

        score = search_result.get("_score")
        if score is None:
            score = search_result.get("@search.score")

        if score is None and search_result.get("distance") is not None:
            score = 1 - search_result["distance"]

        return {
            "blob_name": img_path,
            "similarity_score": score,
            "embedding_model": embedding_model,
            "index_name": index_name,
            "query_face_index": query_face_index,
            "rank": rank,
        }

    def __build_retrievals(
        self,
        search_results: List[List[Dict[str, Any]]],
        index_name: str,
    ) -> List[Dict[str, Any]]:
        retrievals: List[Dict[str, Any]] = []
        for query_face_index, model_results in enumerate(search_results):
            for rank, search_result in enumerate(model_results, start=1):
                match = self.__get_match_result(
                    search_result=search_result,
                    embedding_model=self.embedding_model,
                    index_name=index_name,
                    query_face_index=query_face_index,
                    rank=rank,
                )
                if match is not None:
                    retrievals.append(match)

        return retrievals


class FaceSearch:
    def __init__(
        self,
        vector_db: Indexer,
        project_id: str,
        embedding_models: list,
        model_thresholds: list,
        top_k: Optional[int] = None,
    ):
        self.project_id = project_id
        self.embedding_models = embedding_models
        self.model_thresholds = model_thresholds
        self.top_k = top_k
        self.finders = []
        self.index_names = []
        for i, j in zip(embedding_models, model_thresholds):
            self.finders.append(
                Search(
                    vector_db=vector_db,
                    embedding_model=i,
                    threshold=j,
                    top_k=self.top_k,
                )
            )
            # TODO: abstract index name construction to a function
            self.index_names.append(f"{project_id}_{i.lower()}")

    async def find(self, img_path: str):
        tasks = []
        for finder, index_name in zip(self.finders, self.index_names):
            tasks.append(finder.get_matches(img_path, index_name))
        results = await asyncio.gather(*tasks)
        return list(itertools.chain(*results))
