import asyncio
import itertools
from faceapp._base.indexer import Indexer
from faceapp.utils.processes.extractor import FaceEmbedder


class Search:
    def __init__(self, vector_db: Indexer, embedding_model: str, threshold: float = 0.6):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.face_embedder = FaceEmbedder()

    async def get_matches(self, img_path: str, index_name: str):
        embeddings = self.face_embedder.represent_faces(
            img_path, embedding_model=self.embedding_model, face_detector="mtcnn"
        )["embedding_objs"]
        search_tasks = [
            self.vector_db.search(
                index_name=index_name,
                query_embedding=obj["embedding"],
                threshold=self.threshold,
            )
            for obj in embeddings
        ]
        search_results = await asyncio.gather(*search_tasks)
        search_results_aggregated = list(itertools.chain(*search_results))
        return list(set(list(map(self.__get_blob_url, search_results_aggregated))))

    @staticmethod
    def __get_blob_url(search_result: dict):
        img_path = search_result["blob_name"]
        return img_path

class FaceSearch:
    def __init__(self,vector_db:Indexer, project_id:str, embedding_models:list, model_thresholds:list):
        self.project_id = project_id
        self.embedding_models = embedding_models
        self.model_thresholds = model_thresholds
        self.finders = []
        self.index_names = []
        for i,j in zip(embedding_models, model_thresholds):
            self.finders.append(
                Search(vector_db=vector_db, embedding_model=i, threshold=j)
            )
            # TODO: abstract index name construction to a function
            self.index_names.append(f"{project_id}_{i.lower()}")

    async def find(self, img_path: str):
        tasks = []
        for finder, index_name in zip(self.finders, self.index_names):
            tasks.append(finder.get_matches(img_path, index_name))
        results = await asyncio.gather(*tasks)
        return list(set(list(itertools.chain(*results))))

