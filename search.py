import asyncio
import os

import cv2
from dotenv import load_dotenv

from faceapp.utils.processes.vector_index.chroma_db import ChromadbVectorStore
from faceapp.utils.search import FaceSearch
from utils import load_project_config

PROJECT_NAME = "test"
CONFIG_DIR = "./configs"
config = load_project_config(CONFIG_DIR, PROJECT_NAME)
load_dotenv(config.get("dotenv_path"))


path = "dataset/faces/dad.png"
db_path = "dataset/raw"
embedding_models = config.get("extraction")["embedding_models"]
thresholds = config.get("extraction")["similarity_thresholds"]
project_id = config.get("project_id")

vector_store = ChromadbVectorStore()
finder = FaceSearch(
    vector_db=vector_store,
    embedding_models=embedding_models,
    model_thresholds=thresholds,
    project_id=project_id,
)
results = asyncio.run(finder.find(path))
for result in results:
    if result:
        print(result)
        result = result.split("/")[-1]
        result = os.path.join(db_path, result)
        image = cv2.imread(result)
        cv2.imshow("Output", cv2.imread(result))
        cv2.waitKey(0)
