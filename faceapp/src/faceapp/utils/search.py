import asyncio

from faceapp._base.indexer import Indexer
from faceapp.utils.processes.extractor import FaceEmbedder


class FaceSearch:
    def __init__(self, vector_db: Indexer, embedding_model: str):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        # TODO: Extend search using multiple embedding models on multiple vector indices
        self.face_embedder = FaceEmbedder()

    async def get_matches(self, img_path: str, index_name: str, threshold: float = 0.8):
        embeddings = self.face_embedder.represent_faces(
            img_path, embedding_model=self.embedding_model, face_detector="mtcnn"
        )["embedding_objs"]
        search_tasks = [
            self.vector_db.search(
                index_name=index_name,
                query_embedding=obj["embedding"],
                threshold=threshold,
            )
            for obj in embeddings
        ]
        search_results = await asyncio.gather(*search_tasks)
        search_results_aggregated = []
        [search_results_aggregated.extend(i) for i in search_results]
        return list(set(list(map(self.__get_blob_url, search_results_aggregated))))

    @staticmethod
    def __get_blob_url(search_result: dict):
        img_path = search_result["blob_name"]
        return img_path
