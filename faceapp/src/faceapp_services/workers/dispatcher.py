from __future__ import annotations

"""Task router for worker execution.

The dispatcher maps task types to specialized handlers and always returns a
`WorkerResult` instead of propagating raw exceptions.
"""

from faceapp_services.workers.contracts import (
    ExtractionBatchPayload,
    ExtractionUnitPayload,
    TaskEnvelope,
    TaskType,
    WorkerResult,
)
from faceapp_services.workers.extraction import ExtractionTaskHandler


class WorkerDispatcher:
    """Dispatches typed worker tasks to extraction handlers."""

    def __init__(self):
        self.extraction_handler = ExtractionTaskHandler()

    async def dispatch(self, task: TaskEnvelope) -> WorkerResult:
        """Execute a task envelope.

        Returns normalized success/error output in `WorkerResult`.
        """

        try:
            if task.type == TaskType.EXTRACT_UNIT:
                if not isinstance(task.payload, ExtractionUnitPayload):
                    raise ValueError("Invalid payload type for extract.unit")
                data = await self.extraction_handler.handle_unit(task.payload)
            elif task.type == TaskType.EXTRACT_BATCH:
                if not isinstance(task.payload, ExtractionBatchPayload):
                    raise ValueError("Invalid payload type for extract.batch")
                data = await self.extraction_handler.handle_batch(task.payload)
            else:
                raise ValueError(f"Unsupported task type: {task.type}")

            return WorkerResult(
                task_id=task.id,
                task_type=task.type,
                status="success",
                data=data,
            )
        except Exception as exc:
            return WorkerResult(
                task_id=task.id,
                task_type=task.type,
                status="error",
                error=str(exc),
            )
