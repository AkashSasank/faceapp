import json
from pathlib import Path

from faceapp_services.workers import LocalServerlessWorker, TaskEnvelope


def load_tasks(path: str) -> list[TaskEnvelope]:
    """Load one or many task envelopes from a JSON file."""

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = [payload]
    return [TaskEnvelope.model_validate(item) for item in payload]


def run(task_file: str) -> list[dict]:
    """Run all tasks from a file through the local worker.

    Returns serialized results for each task.
    """

    worker = LocalServerlessWorker()
    tasks = load_tasks(task_file)
    worker.enqueue_many(tasks)
    results = worker.run_all_sync()
    return [result.model_dump() for result in results]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run local worker tasks (serverless-style)."
    )
    parser.add_argument("task_file", help="Path to JSON task or list of tasks")
    args = parser.parse_args()

    output = run(args.task_file)
    print(json.dumps(output, indent=2))
