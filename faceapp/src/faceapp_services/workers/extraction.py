from __future__ import annotations

"""Extraction task handlers used by the worker dispatcher."""

from typing import AsyncGenerator

from faceapp_services.workers.contracts import (
    ExtractionBatchPayload,
    ExtractionUnitPayload,
)

from faceapp._base.base import ProcessOutput
from faceapp.utils.pipelines import LocalImageExtractionPipeline


class ExtractionTaskHandler:
    """Executes extraction tasks using the local image extraction pipeline."""

    def __init__(self):
        self.unit_pipeline = LocalImageExtractionPipeline()

    async def handle_unit(self, payload: ExtractionUnitPayload) -> dict:
        """Run extraction for one path and return serialized output."""

        result = await self.unit_pipeline.ainvoke(
            path=payload.path,
            embedding_models=payload.embedding_models,
            features=payload.features,
            face_detector=payload.face_detector,
        )
        return await self._to_dict(result)

    async def handle_batch(
        self,
        payload: ExtractionBatchPayload,
    ) -> list[dict]:
        """Run extraction sequentially for each path in the batch payload."""

        outputs: list[dict] = []
        for path in payload.paths:
            result = await self.unit_pipeline.ainvoke(
                path=path,
                embedding_models=payload.embedding_models,
                features=payload.features,
                face_detector=payload.face_detector,
            )
            outputs.append(await self._to_dict(result))
        return outputs

    @staticmethod
    async def _to_dict(
        result: ProcessOutput | AsyncGenerator[ProcessOutput, None],
    ) -> dict:
        """Convert process output (direct or streamed) into a dictionary."""

        if isinstance(result, ProcessOutput):
            return result.to_dict()

        async for streamed_result in result:
            return streamed_result.to_dict()

        return {}
        return {}
