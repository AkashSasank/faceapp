from __future__ import annotations

"""Queue abstractions for API producers and worker consumers."""

from typing import Any, Protocol

from faceapp_services.workers.contracts import TaskEnvelope


class TaskQueue(Protocol):
    """Queue interface for enqueue/dequeue operations."""

    def enqueue(self, task: TaskEnvelope) -> None: ...

    def dequeue(self, timeout_seconds: int = 5) -> TaskEnvelope | None: ...


class RedisTaskQueue:
    """Redis list-backed queue implementation."""

    def __init__(self, url: str, queue_name: str = "faceapp:tasks"):
        from redis import Redis

        self.queue_name = queue_name
        self.client = Redis.from_url(url, decode_responses=True)

    def enqueue(self, task: TaskEnvelope) -> None:
        """Push one task envelope to the queue."""

        self.client.lpush(self.queue_name, task.model_dump_json())

    def dequeue(self, timeout_seconds: int = 5) -> TaskEnvelope | None:
        """Pop one task envelope from the queue, blocking for timeout."""

        item: Any = self.client.execute_command(
            "BRPOP",
            self.queue_name,
            timeout_seconds,
        )
        if item is None:
            return None

        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None

        payload = item[1]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        if not isinstance(payload, str):
            return None

        return TaskEnvelope.model_validate_json(payload)


def build_task_queue(
    backend: str,
    url: str,
    queue_name: str,
) -> TaskQueue:
    """Build a queue instance for the selected backend."""

    normalized_backend = backend.strip().lower()
    if normalized_backend == "redis":
        return RedisTaskQueue(url=url, queue_name=queue_name)

    raise ValueError(f"Unsupported queue backend: {backend}")
