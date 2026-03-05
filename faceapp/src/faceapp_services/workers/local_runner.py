from __future__ import annotations

"""Local in-memory worker runtime used for serverless-style simulation."""

import asyncio
from collections import deque
from typing import Deque, Iterable, List

from faceapp_services.workers.contracts import TaskEnvelope, WorkerResult
from faceapp_services.workers.dispatcher import WorkerDispatcher


class LocalServerlessWorker:
    """
    In-memory worker runner that mimics serverless invocation:
    one task envelope in, one result out.
    """

    def __init__(self):
        """Initialize dispatcher and in-memory FIFO task queue."""

        self.dispatcher = WorkerDispatcher()
        self.queue: Deque[TaskEnvelope] = deque()

    def enqueue(self, task: TaskEnvelope) -> None:
        """Push one task envelope into the worker queue."""

        self.queue.append(task)

    def enqueue_many(self, tasks: Iterable[TaskEnvelope]) -> None:
        """Push multiple task envelopes into the worker queue."""

        for task in tasks:
            self.enqueue(task)

    async def run_once(self) -> WorkerResult | None:
        """Process a single queued task and return its result."""

        if not self.queue:
            return None
        task = self.queue.popleft()
        return await self.dispatcher.dispatch(task)

    async def run_all(self) -> List[WorkerResult]:
        """Process all queued tasks until the queue is empty."""

        results: List[WorkerResult] = []
        while self.queue:
            result = await self.run_once()
            if result is not None:
                results.append(result)
        return results

    def run_all_sync(self) -> List[WorkerResult]:
        """Synchronous wrapper around `run_all` for CLI/local scripts."""

        return asyncio.run(self.run_all())
