"""Worker package public exports.

This package provides task contracts and dispatching/execution runtimes.
"""

from typing import TYPE_CHECKING, Any

from faceapp_services.workers import contracts

TaskEnvelope = contracts.TaskEnvelope
TaskType = contracts.TaskType
WorkerResult = contracts.WorkerResult

if TYPE_CHECKING:
    from faceapp_services.workers.dispatcher import WorkerDispatcher
    from faceapp_services.workers.multiprocess_runner import (
        MultiprocessQdrantWorkerRunner,
    )
    from faceapp_services.workers.multiprocess_runner import (
        MultiprocessQdrantWorkerRunner as MultiprocessWorkerRunner,
    )

__all__ = [
    "TaskEnvelope",
    "TaskType",
    "WorkerResult",
    "WorkerDispatcher",
    "MultiprocessWorkerRunner",
    "MultiprocessQdrantWorkerRunner",
]


def __getattr__(name: str) -> Any:
    """Lazily import worker runtime components on demand."""

    if name == "WorkerDispatcher":
        from faceapp_services.workers.dispatcher import WorkerDispatcher

        return WorkerDispatcher

    if name == "MultiprocessWorkerRunner":
        from faceapp_services.workers.multiprocess_runner import (
            MultiprocessQdrantWorkerRunner,
        )

        return MultiprocessQdrantWorkerRunner

    if name == "MultiprocessQdrantWorkerRunner":
        from faceapp_services.workers.multiprocess_runner import (
            MultiprocessQdrantWorkerRunner,
        )

        return MultiprocessQdrantWorkerRunner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
