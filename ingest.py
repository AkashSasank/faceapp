import asyncio
import datetime

from dotenv import load_dotenv

from faceapp.utils.builders import PipelineBuilder
from faceapp.utils.pipelines import (
    AiSearchIndexingPipeline,
    LocalImageExtractionPipeline,
)

load_dotenv(".env")


features = ["age", "gender", "race", "emotion"]


models = [
    # "Facenet",
    # "Facenet512",
    "VGG-Face"
]
extraction_pipeline = LocalImageExtractionPipeline()
indexing_pipeline = AiSearchIndexingPipeline()
builder = PipelineBuilder()
pipeline = (
    builder.add_pipeline(extraction_pipeline)
    .add_pipeline(indexing_pipeline)
    .build("Local Image Extraction")
)


producer_config = {
    "path": "./dataset/raw/DSC00255.jpg",
    "embedding_models": models,
    "features": features,
}

consumer_config = {
    "output_path": "./dataset/processed/",
}

tick = datetime.datetime.now()
asyncio.run(pipeline.ainvoke(**producer_config | consumer_config))
tock = datetime.datetime.now() - tick
print(tock)
