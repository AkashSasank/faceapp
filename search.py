import os

from faceapp._base.indexer import Indexer
from faceapp.utils.vector_index.azure_aisearch import AzureAISearchVectorStore
from faceapp.utils.extractor import FaceEmbedder
from dotenv import load_dotenv
import cv2
load_dotenv(".env")

path = "dataset/raw/IMG_8037.jpg"
db_path = "dataset/raw"
model = "Facenet512"

vector_store = AzureAISearchVectorStore(
    service_name=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
)


class FaceSearch:
    def __init__(self, vector_db: Indexer, embedding_model: str, blob_path: str):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        # TODO: Extend search using multiple embedding models on multiple vector indices
        self.blob_path = blob_path
        self.face_embedder = FaceEmbedder()

    def get_matches(self, img_path: str):
        embeddings = self.face_embedder.represent_faces(
            img_path, embedding_model=self.embedding_model, face_detector="mtcnn"
        )["embedding_objs"]

        results = []
        for obj in embeddings:
            query = obj["embedding"]
            res = vector_store.search(
                index_name=self.embedding_model.lower(),
                query_embedding=query,
                threshold=0.75,
            )
            results.extend(res)
        return list(set(list(map(self.__get_blob_url, results))))

    def __get_blob_url(self, search_result: dict):
        img_path = search_result["blob_name"]
        return os.path.join(self.blob_path, img_path)


finder = FaceSearch(
    vector_db=vector_store, embedding_model=model, blob_path=db_path
)
results = finder.get_matches(path)
for result in results:
    print(result)
    image = cv2.imread(result)
    cv2.imshow("Output", cv2.imread(result))
    cv2.waitKey(0)

