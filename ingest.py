import asyncio
import datetime
import os
from dotenv import load_dotenv

from faceapp.utils.builders import PipelineBuilder
from faceapp.utils.pipelines import (
    AiSearchIndexingPipeline,
    LocalImageExtractionPipeline,
ChromadbIndexingPipeline
)
from faceapp.utils.processes.metadata import ExtractionFormatter

load_dotenv(".env")


features = ["age", "gender", "race", "emotion"]


models = [
    # "Facenet",
    "Facenet512",
    "VGG-Face",
]
extraction_pipeline = LocalImageExtractionPipeline()
# indexing_pipeline = AiSearchIndexingPipeline()
indexing_pipeline = ChromadbIndexingPipeline()
builder = PipelineBuilder()
pipeline = (
    builder.add_pipeline(extraction_pipeline)
    .add_pipeline(indexing_pipeline)
    .build("Local Image Extraction")
)

dir = "./dataset/test"

for img in os.listdir(dir):
    path = os.path.join(dir, img)
    producer_config = {
        "path": path,
        "embedding_models": models,
        "features": features,
        "project_id": "hhgdgttstsggcosine",
    }

    consumer_config = {
        "output_path": "./dataset/processed/",
    }

    tick = datetime.datetime.now()
    try:
        out = asyncio.run(pipeline.ainvoke(**producer_config | consumer_config))
        print(out.keys())
    except Exception as e:
        raise e
    tock = datetime.datetime.now() - tick
    print(tock)
