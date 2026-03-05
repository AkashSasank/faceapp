from __future__ import annotations

"""Multiprocess worker runner.

Each worker process consumes task envelopes from a shared queue, executes the
local dispatcher, and returns normalized task results.
"""

import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Iterable

from faceapp_services.workers.contracts import TaskEnvelope, WorkerResult
from faceapp_services.workers.dispatcher import WorkerDispatcher


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

        task = TaskEnvelope.model_validate(task_data)
        result = asyncio.run(dispatcher.dispatch(task))
        result_queue.put(result.model_dump())


class MultiprocessQdrantWorkerRunner:
    """Run many worker processes.

    Each process consumes tasks and executes configured worker strategies.
    """

    def __init__(
        self,
        num_workers: int = 2,
    ):
        """Initialize runner with the number of process workers to launch."""

        self.num_workers = num_workers

    def run(self, tasks: Iterable[TaskEnvelope]) -> list[WorkerResult]:
        """Execute all tasks across multiple worker processes."""

        ctx = mp.get_context("spawn")
        task_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        task_list = list(tasks)
        for task in task_list:
            task_queue.put(task.model_dump())

        processes: list = []
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

        for process in processes:
            process.join()

        return results
