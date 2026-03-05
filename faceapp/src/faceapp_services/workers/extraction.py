from __future__ import annotations

"""Extraction task handlers used by the worker dispatcher."""

from typing import AsyncGenerator

from faceapp_services.workers.contracts import (
    ExtractionBatchPayload,
    ExtractionUnitPayload,
)
from faceapp_services.workers.strategies import WorkerStrategyResolver

from faceapp._base.base import ProcessOutput
from faceapp._base.pipeline import Pipeline


class ExtractionTaskHandler:
    """Executes extraction and indexing tasks using configured strategies."""

    def __init__(self):
        self.strategy_resolver = WorkerStrategyResolver()

    async def handle_unit(self, payload: ExtractionUnitPayload) -> dict:
        """Run extraction and indexing for one image path."""

        strategies = self.strategy_resolver.resolve(
            project_id=payload.project_id,
        )
        extraction_pipeline = strategies.extraction.build_pipeline()
        indexing_pipeline = strategies.indexing.build_pipeline()

        result = await extraction_pipeline.ainvoke(
            path=payload.path,
            embedding_models=payload.embedding_models,
            features=payload.features,
            face_detector=payload.face_detector,
        )
        extraction_output = await self._to_dict(result)
        indexing_output = await self._index_extractions(
            extraction_output=extraction_output,
            indexing_pipeline=indexing_pipeline,
            project_id=payload.project_id,
            index_config=strategies.index_config,
        )
        return {
            "extraction": extraction_output,
            "indexing": indexing_output,
        }

    async def handle_batch(
        self,
        payload: ExtractionBatchPayload,
    ) -> list[dict]:
        """Run extraction and indexing sequentially for each path in batch."""

        strategies = self.strategy_resolver.resolve(
            project_id=payload.project_id,
        )
        extraction_pipeline = strategies.extraction.build_pipeline()
        indexing_pipeline = strategies.indexing.build_pipeline()

        outputs: list[dict] = []
        for path in payload.paths:
            result = await extraction_pipeline.ainvoke(
                path=path,
                embedding_models=payload.embedding_models,
                features=payload.features,
                face_detector=payload.face_detector,
            )
            extraction_output = await self._to_dict(result)
            indexing_output = await self._index_extractions(
                extraction_output=extraction_output,
                indexing_pipeline=indexing_pipeline,
                project_id=payload.project_id,
                index_config=strategies.index_config,
            )
            outputs.append(
                {
                    "extraction": extraction_output,
                    "indexing": indexing_output,
                }
            )
        return outputs

    async def _index_extractions(
        self,
        extraction_output: dict,
        indexing_pipeline: Pipeline,
        project_id: str | None,
        index_config: dict,
    ) -> dict:
        """Index extracted faces using the resolved indexing pipeline."""

        extractions = extraction_output.get("extractions", [])
        if not isinstance(extractions, list) or not extractions:
            return {}

        result = await indexing_pipeline.ainvoke(
            extractions=extractions,
            project_id=project_id,
            index_config=index_config,
        )
        return await self._to_dict(result)

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
