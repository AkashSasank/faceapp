import asyncio
import os

import cv2
from dotenv import load_dotenv

from faceapp.utils.processes.vector_index.chroma_db import ChromadbVectorStore
from faceapp.utils.search import FaceSearch

load_dotenv(".env")



path = "amm.png"
db_path = "dataset/test"
embedding_models = [
    "Facenet512",
    "VGG-Face",
    "DeepID"
]
thresholds = [
    0.4, 0.5, 0.05
]

vector_store = ChromadbVectorStore()
project_id = "hhgdgttstsgsgsgggcosine"

finder = FaceSearch(vector_db=vector_store,
                    embedding_models=embedding_models,
                    model_thresholds=thresholds,
                    project_id=project_id)
results = asyncio.run(finder.find(path))
for result in results:
    print(result)
    result = result.split("/")[-1]
    result = os.path.join(db_path, result)
    image = cv2.imread(result)
    cv2.imshow("Output", cv2.imread(result))
    cv2.waitKey(0)
