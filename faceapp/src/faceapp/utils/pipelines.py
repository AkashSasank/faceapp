import os
from typing import AsyncGenerator

from faceapp._base.pipeline import Pipeline
from faceapp.utils.extractor import FaceExtractor
from faceapp.utils.fetcher import LocalImageFetcher
from faceapp.utils.metadata import LocalImageExtractionFormatter
from faceapp.utils.vector_index import ElasticVectorStore
from faceapp.utils.cleanup import FileCleanup


class LocalImageExtractionPipeline(Pipeline):

    def __init__(self):
        processes = {
            "image_fetcher": LocalImageFetcher(),
            "face_extraction": FaceExtractor(),
        }
        super(LocalImageExtractionPipeline, self).__init__(processes)

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


class LocalImageIndexingPipeline(Pipeline):
    def __init__(self):
        processes = {
            "formatter": LocalImageExtractionFormatter(),
            "vector_index": ElasticVectorStore(),
            "cleanup": FileCleanup(),
        }
        super(LocalImageIndexingPipeline, self).__init__(processes)

    async def ainvoke(self, extractions: list, output_path: str):
        return await super().ainvoke(extractions=extractions, output_path=output_path)
