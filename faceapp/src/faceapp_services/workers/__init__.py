"""Worker package public exports.

This package provides task contracts, dispatching, and a local in-memory
worker runner that mimics serverless invocation style.
"""

from typing import Any

from faceapp_services.workers import contracts

TaskEnvelope = contracts.TaskEnvelope
TaskType = contracts.TaskType
WorkerResult = contracts.WorkerResult

__all__ = [
    "TaskEnvelope",
    "TaskType",
    "WorkerResult",
    "TaskQueue",
    "RedisTaskQueue",
    "build_task_queue",
    "WorkerDispatcher",
    "LocalServerlessWorker",
    "MultiprocessQdrantWorkerRunner",
]


def __getattr__(name: str) -> Any:
    """Lazily import worker runtime components on demand."""

    if name == "TaskQueue":
        from faceapp_services.workers.queue import TaskQueue

        return TaskQueue

    if name == "RedisTaskQueue":
        from faceapp_services.workers.queue import RedisTaskQueue

        return RedisTaskQueue

    if name == "build_task_queue":
        from faceapp_services.workers.queue import build_task_queue

        return build_task_queue

    if name == "WorkerDispatcher":
        from faceapp_services.workers.dispatcher import WorkerDispatcher

        return WorkerDispatcher

    if name == "LocalServerlessWorker":
        from faceapp_services.workers.local_runner import LocalServerlessWorker

        return LocalServerlessWorker

    if name == "MultiprocessQdrantWorkerRunner":
        from faceapp_services.workers.multiprocess_runner import (
            MultiprocessQdrantWorkerRunner,
        )

        return MultiprocessQdrantWorkerRunner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
