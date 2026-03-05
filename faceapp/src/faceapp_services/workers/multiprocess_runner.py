from __future__ import annotations

"""Multiprocess worker runner.

Each worker process consumes task envelopes from a shared queue, executes the
local dispatcher, and writes results to Qdrant.
"""

import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Iterable, Optional

from faceapp_services.workers.contracts import TaskEnvelope, WorkerResult
from faceapp_services.workers.dispatcher import WorkerDispatcher
from faceapp_services.workers.result_store import QdrantWorkerResultStore


def _worker_loop(
    worker_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_path: Optional[str],
    result_collection: str,
):
    dispatcher = WorkerDispatcher()
    store = QdrantWorkerResultStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        path=qdrant_path,
        collection_name=result_collection,
    )

    while True:
        task_data = task_queue.get()
        if task_data is None:
            break

        task = TaskEnvelope.model_validate(task_data)
        result = asyncio.run(dispatcher.dispatch(task))
        store.write_result(result=result, worker_id=worker_id)
        result_queue.put(result.model_dump())


class MultiprocessQdrantWorkerRunner:
    """Run many worker processes.

    Each process consumes tasks and writes results to Qdrant.
    """

    def __init__(
        self,
        num_workers: int = 2,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        result_collection: str = "faceapp_worker_results",
    ):
        self.num_workers = num_workers
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_path = qdrant_path
        self.result_collection = result_collection

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
                args=(
                    f"worker-{index + 1}",
                    task_queue,
                    result_queue,
                    self.qdrant_url,
                    self.qdrant_api_key,
                    self.qdrant_path,
                    self.result_collection,
                ),
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
