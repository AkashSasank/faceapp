import asyncio
import datetime

from faceapp.manager import PipelineManager
from faceapp.utils.pipelines import (
    LocalImageExtractionPipeline,
    LocalImageIndexingPipeline,
)

from dotenv import load_dotenv
load_dotenv(".env")


features = ["age", "gender", "race", "emotion"]


models = [
    # "Facenet",
    "Facenet512",
]
extraction_pipeline = LocalImageExtractionPipeline()
indexing_pipeline = LocalImageIndexingPipeline()
manager = PipelineManager(
    producer_pipeline=extraction_pipeline, consumer_pipeline=indexing_pipeline
)


producer_config = {
    "path": "./dataset/raw/",
    "embedding_models": models,
    "features": features,
}

consumer_config = {
    "output_path": "./dataset/processed/",
}

tick = datetime.datetime.now()
asyncio.run(
    manager.run(producer_config=producer_config, consumer_config=consumer_config)
)
tock = datetime.datetime.now() - tick
print(tock)
