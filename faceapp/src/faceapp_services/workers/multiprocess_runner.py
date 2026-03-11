from __future__ import annotations

"""Multiprocess worker runner.

Each worker process consumes task envelopes from a shared queue, executes the
local dispatcher, and returns normalized task results.
"""

import asyncio
import gc
import multiprocessing as mp
from queue import Empty
from typing import Any, Iterable

from faceapp_services.workers.contracts import TaskEnvelope, TaskType, WorkerResult
from faceapp_services.workers.dispatcher import WorkerDispatcher


def _normalize_queue_value(value: Any) -> Any:
    """Convert non-primitive values recursively for process-safe payloads."""
    if isinstance(value, dict):
        return {key: _normalize_queue_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_queue_value(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            return _normalize_queue_value(converted)
        except (TypeError, ValueError):
            pass

    return value


def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Worker process loop that consumes, dispatches, and emits results."""

    dispatcher = WorkerDispatcher()

    while True:
        task_data = task_queue.get()
        if task_data is None:
            break

        task: TaskEnvelope | None = None
        result: WorkerResult | None = None
        try:
            task = TaskEnvelope.model_validate(task_data)
            result = asyncio.run(dispatcher.dispatch(task))
        except Exception as exc:
            raw_task_id = "unknown"
            raw_task_type = TaskType.EXTRACT_UNIT
            if isinstance(task_data, dict):
                maybe_task_id = task_data.get("id")
                if isinstance(maybe_task_id, str) and maybe_task_id.strip():
                    raw_task_id = maybe_task_id

                maybe_task_type = task_data.get("type")
                try:
                    raw_task_type = TaskType(maybe_task_type)
                except Exception:
                    pass

            result = WorkerResult(
                task_id=raw_task_id,
                task_type=raw_task_type,
                status="error",
                error=str(exc),
            )

        result_queue.put(_normalize_queue_value(result.model_dump()))
        del task_data
        if task is not None:
            del task
        if result is not None:
            del result
        gc.collect()


class MultiprocessQdrantWorkerRunner:
    """Run many worker processes.

    Each process consumes tasks and executes configured worker strategies.
    """

    def __init__(
        self,
        num_workers: int = 2,
    ):
        """Initialize runner with the number of process workers to launch."""

        self.num_workers = max(int(num_workers), 1)

    def run(self, tasks: Iterable[TaskEnvelope]) -> list[WorkerResult]:
        """Execute all tasks across multiple worker processes."""

        ctx = mp.get_context("spawn")
        task_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        task_list = list(tasks)
        if not task_list:
            return []

        for task in task_list:
            task_queue.put(_normalize_queue_value(task.model_dump()))

        processes: list = []
        try:
            for index in range(self.num_workers):
                task_queue.put(None)
                process = ctx.Process(
                    target=_worker_loop,
                    args=(task_queue, result_queue),
                )
                process.start()
                processes.append(process)

            results: list[WorkerResult] = []
            expected = len(task_list)
            while len(results) < expected:
                try:
                    result_data = result_queue.get(timeout=60)
                except Empty as exc:
                    message = "Timed out waiting for worker results"
                    raise RuntimeError(message) from exc
                results.append(WorkerResult.model_validate(result_data))

            return results
        finally:
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

            task_queue.close()
            result_queue.close()
