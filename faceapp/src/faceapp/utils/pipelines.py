import os

from faceapp._base.base import ProcessOutput
from faceapp._base.pipeline import Pipeline
from faceapp.utils.misc import is_valid_image
from faceapp.utils.processes.extractor import FaceExtractor
from faceapp.utils.processes.fetcher import LocalImageFetcher
from faceapp.utils.processes.metadata import ExtractionFormatter
from faceapp.utils.processes.vector_index.qdrant import QdrantVectorStore


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
        self,
        path: str,
        embedding_models: list,
        features: list,
        *args,
        **kwargs,
    ):
        return await self._call_pipeline(
            path=path,
            embedding_models=embedding_models,
            features=features,
            *args,
            **kwargs,
        )

    async def _call_pipeline(
        self,
        path: str,
        embedding_models: list,
        features: list,
        *args,
        **kwargs,
    ):
        if is_valid_image(path):
            data = await super().ainvoke(
                path=path,
                embedding_models=embedding_models,
                features=features,
                *args,
                **kwargs,
            )
            return data
        return ProcessOutput.model_validate({})


class QdrantIndexingPipeline(Pipeline):
    """
    Pipeline to format face extractions and upload them to Qdrant
    """

    def __init__(self, name: str = "QdrantIndexingPipeline"):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_path = os.getenv("QDRANT_PATH")
        processes = {
            "formatter": ExtractionFormatter(),
            "vector_index": QdrantVectorStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                path=qdrant_path,
            ),
        }

        super(QdrantIndexingPipeline, self).__init__(processes, name)
