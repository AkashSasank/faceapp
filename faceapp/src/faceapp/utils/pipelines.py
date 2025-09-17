import os
from typing import AsyncGenerator

from faceapp._base.pipeline import Pipeline
from faceapp.utils.processes.extractor import FaceExtractor
from faceapp.utils.processes.fetcher import LocalImageFetcher
from faceapp.utils.processes.metadata import (
    ExtractionFormatter,
)
from faceapp.utils.processes.vector_index.azure_aisearch import AzureAISearchVectorStore


class LocalImageExtractionPipeline(Pipeline):
    """
    Extracts faces from a single image
    """

    def __init__(self, name: str = "LocalImageExtractionPipeline"):
        processes = {
            "image_fetcher": LocalImageFetcher(),
            "face_extraction": FaceExtractor(),
        }
        super(LocalImageExtractionPipeline, self).__init__(processes, name)

    async def ainvoke(self, path: str, embedding_models: list, features: list) -> dict:
        return await self.__call_pipeline(
            path=path, embedding_models=embedding_models, features=features
        )

    async def __call_pipeline(
        self, path: str, embedding_models: list, features: list
    ) -> dict:
        extension = path.split(".")[-1].lower()
        if extension in ["jpg", "jpeg", "png"]:
            data = await super().ainvoke(
                path=path,
                embedding_models=embedding_models,
                features=features,
            )
            return data
        return {}


class LocalImageDirExtractionPipeline(LocalImageExtractionPipeline):
    """
    Extracts faces from a local dir
    """

    async def ainvoke(
        self, path: str, embedding_models: list, features: list
    ) -> AsyncGenerator[dict, None]:
        if os.path.isdir(path):
            for image in os.listdir(path):
                img_path = os.path.join(path, image)
                output = await self.__call_pipeline(
                    img_path, embedding_models, features
                )
                yield output
        else:
            yield await self.__call_pipeline(
                path=path, embedding_models=embedding_models, features=features
            )


class AiSearchIndexingPipeline(Pipeline):
    def __init__(self, name: str = "AiSearchIndexingPipeline"):
        processes = {
            "formatter": ExtractionFormatter(),
            "vector_index": AzureAISearchVectorStore(
                service_name=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
                api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
            ),
        }
        super(AiSearchIndexingPipeline, self).__init__(processes, name)

    async def ainvoke(self, extractions: list, image_metadata: dict, **kwargs):
        return await super().ainvoke(
            extractions=extractions, image_metadata=image_metadata, **kwargs
        )
