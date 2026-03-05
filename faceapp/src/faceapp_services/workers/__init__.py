"""Worker package public exports.

This package provides task contracts, dispatching, and a local in-memory
worker runner that mimics serverless invocation style.
"""

from faceapp_services.workers import contracts, multiprocess_runner
from faceapp_services.workers.dispatcher import WorkerDispatcher
from faceapp_services.workers.local_runner import LocalServerlessWorker

TaskEnvelope = contracts.TaskEnvelope
TaskType = contracts.TaskType
WorkerResult = contracts.WorkerResult
MultiprocessQdrantWorkerRunner = multiprocess_runner.MultiprocessQdrantWorkerRunner

__all__ = [
    "LocalServerlessWorker",
    "MultiprocessQdrantWorkerRunner",
    "TaskEnvelope",
    "TaskType",
    "WorkerDispatcher",
    "WorkerResult",
]
