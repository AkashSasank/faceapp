import os

from faceapp._base.base import ProcessOutput
from faceapp._base.pipeline import Pipeline
from faceapp.utils.misc import is_valid_image
from faceapp.utils.processes.extractor import FaceExtractor
from faceapp.utils.processes.fetcher import LocalImageFetcher, S3ImageFetcher
from faceapp.utils.processes.metadata import ExtractionFormatter
from faceapp.utils.processes.vector_index.azure_aisearch import AzureAISearchVectorStore
from faceapp.utils.processes.vector_index.chroma_db import ChromadbVectorStore


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

    async def ainvoke(
        self, path: str, embedding_models: list, features: list, *args, **kwargs
    ):
        return await self._call_pipeline(
            path=path,
            embedding_models=embedding_models,
            features=features,
            *args,
            **kwargs
        )

    async def _call_pipeline(
        self, path: str, embedding_models: list, features: list, *args, **kwargs
    ):
        if is_valid_image(path):
            data = await super().ainvoke(
                path=path,
                embedding_models=embedding_models,
                features=features,
                *args,
                **kwargs
            )
            return data
        return ProcessOutput.model_validate({})


class LocalImageDirExtractionPipeline(LocalImageExtractionPipeline):
    """
    Extracts faces from a local dir
    """

    async def ainvoke(
        self, path: str, embedding_models: list, features: list, *args, **kwargs
    ):
        if os.path.isdir(path):
            for image in os.listdir(path):
                img_path = os.path.join(path, image)
                output = await self._call_pipeline(
                    img_path, embedding_models, features, *args, **kwargs
                )
                yield output
        else:
            yield await self._call_pipeline(
                path=path,
                embedding_models=embedding_models,
                features=features,
                *args,
                **kwargs
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

    async def ainvoke(self, extractions: list, **kwargs):
        return await super().ainvoke(extractions=extractions, **kwargs)


class S3ImageExtractorPipeline(Pipeline):
    """
    Pipeline to fetch image from s3 bucket and do face extraction
    """

    def __init__(self, name: str = "S3ImageExtractorPipeline"):
        processes = {
            "image_fetcher": S3ImageFetcher(),
            "extractor": FaceExtractor(),
        }
        super(S3ImageExtractorPipeline, self).__init__(processes, name)


class ChromadbIndexingPipeline(Pipeline):
    """
    Pipeline to format face extractions and upload them to chromaDB
    """

    def __init__(self, name: str = "ChromadbIndexingPipeline"):
        processes = {
            "formatter": ExtractionFormatter(),
            "vector_index": ChromadbVectorStore(),
        }

        super(ChromadbIndexingPipeline, self).__init__(processes, name)
