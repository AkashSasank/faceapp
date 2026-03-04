import asyncio
import datetime
import os

from dotenv import load_dotenv

from faceapp.utils.builders import PipelineBuilder
from faceapp.utils.pipelines import (ChromadbIndexingPipeline,
                                     LocalImageExtractionPipeline)
from utils import load_config

PROJECT_NAME = "foo"
CONFIG_FILE_NAME = "chroma.yaml"

ingest_config = load_config(f"./configs/{CONFIG_FILE_NAME}", PROJECT_NAME)
load_dotenv(ingest_config.get("dotenv_path"))

config = {
    "project_id": ingest_config["project_id"],
    "features": ingest_config["extraction"].get("features", []),
    "embedding_models": ingest_config["extraction"].get("embedding_models", []),
    "output_path": ingest_config["files"].get("output_path"),
    "index_config": ingest_config.get("index_config"),
}
# TODO: Handle failed extractions, implement retry
# TODO: Move processed images to output folder
# TODO: Add logger
extraction_pipeline = LocalImageExtractionPipeline()
indexing_pipeline = ChromadbIndexingPipeline()
builder = PipelineBuilder()
pipeline = (
    builder.add_pipeline(extraction_pipeline)
    .add_pipeline(indexing_pipeline)
    .build("Local Image Extraction")
)

input_paths = ingest_config["files"]["input_path"]
if isinstance(input_paths, str):
    input_paths = [input_paths]

for input_dir in input_paths:
    for img in os.listdir(input_dir):
        path = os.path.join(input_dir, img)
        step_config = config | {"path": path}
        tick = datetime.datetime.now()

        try:
            out = asyncio.run(pipeline.ainvoke(**step_config))
        except Exception as e:
            print(e)
            pass

        tock = datetime.datetime.now() - tick
        print(tock)
