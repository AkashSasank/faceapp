import json
from pathlib import Path

import faceapp_services.workers as worker_pkg


def load_tasks(path: str) -> list[worker_pkg.TaskEnvelope]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = [payload]
    return [worker_pkg.TaskEnvelope.model_validate(item) for item in payload]


def run(task_file: str, workers: int = 2) -> list[dict]:
    runner = worker_pkg.MultiprocessQdrantWorkerRunner(num_workers=workers)
    tasks = load_tasks(task_file)
    results = runner.run(tasks)
    return [result.model_dump() for result in results]


if __name__ == "__main__":
    import argparse

    description_text = (
        "Run multiprocess worker tasks and persist results to Qdrant."
    )
    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument("task_file", help="Path to JSON task or list of tasks")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output = run(task_file=args.task_file, workers=args.workers)
    print(json.dumps(output, indent=2))
