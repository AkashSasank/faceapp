from __future__ import annotations

"""Management commands for local development.

Usage examples:
- faceapp run server --host 0.0.0.0 --port 8000
- faceapp run workers --task-file configs/worker_tasks.sample.json
- faceapp run all --task-file configs/worker_tasks.sample.json
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import uvicorn

if TYPE_CHECKING:
    from faceapp_services.workers import TaskEnvelope


def _load_tasks(task_file: str) -> list[TaskEnvelope]:
    """Load task envelopes from a JSON file path."""
    from faceapp_services.workers import TaskEnvelope

    payload = json.loads(Path(task_file).read_text())
    if isinstance(payload, dict):
        payload = [payload]
    return [TaskEnvelope.model_validate(item) for item in payload]


def _run_local_workers(tasks: Iterable[TaskEnvelope]) -> list[dict]:
    """Execute tasks using the in-process local worker implementation."""
    from faceapp_services.workers import LocalServerlessWorker

    worker = LocalServerlessWorker()
    worker.enqueue_many(tasks)
    results = worker.run_all_sync()
    return [result.model_dump() for result in results]


def _run_multiprocess_workers(
    tasks: list[TaskEnvelope],
    workers: int,
) -> list[dict]:
    """Execute tasks using the multiprocess worker runner."""
    from faceapp_services.workers import MultiprocessQdrantWorkerRunner

    runner = MultiprocessQdrantWorkerRunner(num_workers=workers)
    results = runner.run(tasks)
    return [result.model_dump() for result in results]


def runworkers_command(args: argparse.Namespace) -> int:
    """Run worker tasks once or in watch mode and print JSON results."""
    seen_task_ids: set[str] = set()

    while True:
        tasks = _load_tasks(args.task_file)
        pending_tasks: list[TaskEnvelope] = []
        for task in tasks:
            if task.id not in seen_task_ids:
                pending_tasks.append(task)

        if pending_tasks:
            if args.mode == "local":
                results = _run_local_workers(pending_tasks)
            else:
                results = _run_multiprocess_workers(
                    tasks=pending_tasks,
                    workers=args.workers,
                )

            for task in pending_tasks:
                seen_task_ids.add(task.id)

            print(json.dumps(results, indent=2))

        if not args.watch:
            break

        time.sleep(args.poll_interval)

    return 0


def runserver_command(args: argparse.Namespace) -> int:
    """Start the FastAPI application with uvicorn."""
    uvicorn.run(
        "faceapp_services.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _runserver_process(host: str, port: int, reload: bool):
    """Target function for running the API server in a child process."""
    uvicorn.run(
        "faceapp_services.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def _runworkers_process(
    mode: str,
    task_file: str,
    workers: int,
    watch: bool,
    poll_interval: float,
):
    """Target function for running workers in a child process."""
    args = argparse.Namespace(
        mode=mode,
        task_file=task_file,
        workers=workers,
        watch=watch,
        poll_interval=poll_interval,
    )
    runworkers_command(args)


def runall_command(args: argparse.Namespace) -> int:
    """Start API server and workers concurrently as separate processes."""
    ctx = mp.get_context("spawn")

    api_process = ctx.Process(
        target=_runserver_process,
        args=(args.host, args.port, False),
    )
    worker_process = ctx.Process(
        target=_runworkers_process,
        args=(
            args.mode,
            args.task_file,
            args.workers,
            args.watch,
            args.poll_interval,
        ),
    )

    api_process.start()
    worker_process.start()

    try:
        api_process.join()
        worker_process.join()
    except KeyboardInterrupt:
        if api_process.is_alive():
            api_process.terminate()
        if worker_process.is_alive():
            worker_process.terminate()
        api_process.join()
        worker_process.join()

    return 0


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    """Register shared server arguments on a command parser."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def _add_worker_args(parser: argparse.ArgumentParser) -> None:
    """Register shared worker arguments on a command parser."""
    parser.add_argument(
        "--mode",
        choices=["local", "multiprocess"],
        default="multiprocess",
    )
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the root CLI parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="FaceApp management commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    runserver = subparsers.add_parser("runserver", help="Start FastAPI server")
    _add_server_args(runserver)
    runserver.add_argument("--reload", action="store_true")
    runserver.set_defaults(handler=runserver_command)

    runworkers = subparsers.add_parser(
        "runworkers",
        help="Run worker consumers",
    )
    _add_worker_args(runworkers)
    runworkers.set_defaults(handler=runworkers_command)

    runall = subparsers.add_parser(
        "runall",
        help="Start API server and workers together",
    )
    _add_server_args(runall)
    _add_worker_args(runall)
    runall.set_defaults(handler=runall_command)

    run = subparsers.add_parser("run", help="Run grouped commands")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)

    run_server = run_subparsers.add_parser(
        "server",
        help="Start FastAPI server",
    )
    _add_server_args(run_server)
    run_server.add_argument("--reload", action="store_true")
    run_server.set_defaults(handler=runserver_command)

    run_workers = run_subparsers.add_parser(
        "workers",
        help="Run worker consumers",
    )
    _add_worker_args(run_workers)
    run_workers.set_defaults(handler=runworkers_command)

    run_all = run_subparsers.add_parser(
        "all",
        help="Start API server and workers together",
    )
    _add_server_args(run_all)
    _add_worker_args(run_all)
    run_all.set_defaults(handler=runall_command)

    return parser


def main() -> int:
    """CLI entry point for the `faceapp` command."""
    parser = _build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
