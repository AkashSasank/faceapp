import asyncio
import os

import cv2
from dotenv import load_dotenv

from faceapp.utils.processes.vector_index.azure_aisearch import AzureAISearchVectorStore
from faceapp.utils.search import FaceSearch

load_dotenv(".env")

path = "jd.png"
db_path = "dataset/test"
model = "VGG-Face"

vector_store = AzureAISearchVectorStore(
    service_name=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
)
index_name = "hhgdgttstsggcosine_vgg-face"

finder = FaceSearch(vector_db=vector_store, embedding_model=model)
results = asyncio.run(finder.get_matches(path, index_name=index_name, threshold=0.70))
for result in results:
    print(result)
    result = result.split("/")[-1]
    result = os.path.join(db_path, result)
    image = cv2.imread(result)
    cv2.imshow("Output", cv2.imread(result))
    cv2.waitKey(0)
