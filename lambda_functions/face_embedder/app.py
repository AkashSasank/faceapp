import asyncio
import json
import os

from faceapp.utils.builders import PipelineBuilder
from faceapp.utils.pipelines import AzureAISearchVectorStore, S3ImageExtractorPipeline
from faceapp.utils.processes.cleanup import S3Cleanup

pipeline = (
    PipelineBuilder()
    .add_pipeline(S3ImageExtractorPipeline, name="S3ImageExtractor")
    .add_pipeline(AzureAISearchVectorStore, name="AzureAISearchVectorStore")
    .add_pipeline(S3Cleanup, name="S3Cleanup")
    .build(name="S3ImageExtractorPipeline")
)

features = ["age", "gender", "race", "emotion"]


models = [
    # "Facenet",
    # "Facenet512",
    "VGG-Face"
]
output_bucket = os.environ.get("OUTPUT_BUCKET")


def create_config(event):
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    config = {
        "input_bucket": bucket_name,
        "blob_name": key,
        "download_dir": "/tmp",
        "embedding_models": models,
        "features": features,
        "output_bucket": output_bucket,
    }
    return config


def handler(event, context):
    print(event, context)
    config = create_config(event)
    print(config)
    status = asyncio.run(pipeline.ainvoke(**config))
    print(status)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Success", "event": event}),
    }
