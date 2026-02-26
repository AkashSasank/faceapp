import asyncio
import datetime
import os
import shutil
from dotenv import load_dotenv

from faceapp.utils.builders import PipelineBuilder
from faceapp.utils.pipelines import (
    LocalImageExtractionPipeline,
ChromadbIndexingPipeline
)
from utils import load_config

PROJECT_NAME = "foo"
CONFIG_FILE_NAME = "chroma.yaml"

ingest_config = load_config(
    f"./configs/{CONFIG_FILE_NAME}",
                            PROJECT_NAME)
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

failed_path = ingest_config["files"].get("failed_path")
if failed_path and isinstance(failed_path, list):
    failed_path = failed_path[0]
if failed_path and not os.path.exists(failed_path):
    os.makedirs(failed_path)

for input_dir in input_paths:
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        continue
    for img in os.listdir(input_dir):
        path = os.path.join(input_dir, img)
        step_config = config | {"path": path}
        tick = datetime.datetime.now()

        try:
            print(f"Processing {img}...")
            out = asyncio.run(pipeline.ainvoke(**step_config))
        except Exception as e:
            print(f"Error processing {path}: {e}")
            if failed_path:
                try:
                    shutil.move(path, os.path.join(failed_path, img))
                    print(f"Moved {img} to {failed_path}")
                except Exception as move_error:
                    print(f"Failed to move {img}: {move_error}")
            continue

        tock = datetime.datetime.now() - tick
        print(f"Processed {img} in {tock}")
