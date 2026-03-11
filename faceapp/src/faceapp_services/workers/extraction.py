from __future__ import annotations

"""Extraction task handlers used by the worker dispatcher."""

import gc
import os
from typing import AsyncGenerator

from faceapp_services.workers.contracts import (
    ExtractionBatchPayload,
    ExtractionUnitPayload,
)
from faceapp_services.workers.runtime import (
    compact_extraction_output,
    get_process_memory_snapshot,
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

        return await self._process_path(
            path=payload.path,
            embedding_models=payload.embedding_models,
            features=payload.features,
            face_detector=payload.face_detector,
            meta=payload.meta,
            project_id=payload.project_id,
            extraction_pipeline=extraction_pipeline,
            indexing_pipeline=indexing_pipeline,
            index_config=strategies.index_config,
        )

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
            outputs.append(
                await self._process_path(
                    path=path,
                    embedding_models=payload.embedding_models,
                    features=payload.features,
                    face_detector=payload.face_detector,
                    meta=payload.meta,
                    project_id=payload.project_id,
                    extraction_pipeline=extraction_pipeline,
                    indexing_pipeline=indexing_pipeline,
                    index_config=strategies.index_config,
                )
            )
        return outputs

    async def _process_path(
        self,
        *,
        path: str,
        embedding_models: list[str],
        features: list[str],
        face_detector: str,
        meta: dict,
        project_id: str | None,
        extraction_pipeline: Pipeline,
        indexing_pipeline: Pipeline,
        index_config: dict,
    ) -> dict:
        """Extract and index one path while keeping result payloads small."""

        memory_before_extract = get_process_memory_snapshot()
        result = await extraction_pipeline.ainvoke(
            path=path,
            embedding_models=embedding_models,
            features=features,
            face_detector=face_detector,
            meta=meta,
        )
        extraction_output = await self._to_dict(result)
        memory_after_extract = get_process_memory_snapshot()
        indexing_output = await self._index_extractions(
            extraction_output=extraction_output,
            indexing_pipeline=indexing_pipeline,
            project_id=project_id,
            index_config=index_config,
        )
        compacted_extraction = compact_extraction_output(extraction_output)
        del result
        del extraction_output
        gc.collect()

        return {
            "extraction": compacted_extraction,
            "indexing": indexing_output,
            "runtime": {
                "worker_pid": os.getpid(),
                "memory": {
                    "before_extract": memory_before_extract,
                    "after_extract": memory_after_extract,
                    "after_index": get_process_memory_snapshot(),
                },
            },
        }

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
