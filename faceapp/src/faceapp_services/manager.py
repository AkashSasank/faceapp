from __future__ import annotations

"""Management commands for local development.

Usage examples:
- faceapp run server --host 0.0.0.0 --port 8000
- faceapp runworkers --source queue --workers 3
- faceapp run all --task-file configs/worker_tasks.sample.json
- faceapp ingest --file ./dataset/raw/sample.jpg --config-name foo
- faceapp ingest --folder ./dataset/raw --config-name foo
- faceapp search --file ./dataset/raw/query.jpg --config-name foo
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import ulid
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

    if args.source == "queue":
        return _run_queue_consumers(args)

    if not args.task_file:
        raise ValueError("--task-file is required when --source=task-file")

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


def _queue_consumer_loop(
    queue_backend: str,
    queue_url: str,
    queue_name: str,
    poll_interval: float,
):
    """Continuously consume queue tasks and dispatch them synchronously.

    Each consumed task is executed through the local dispatcher and printed as
    JSON for easy log inspection in local environments.
    """

    from faceapp_services.workers import WorkerDispatcher
    from faceapp_services.workers.queue import build_task_queue

    queue = build_task_queue(
        backend=queue_backend,
        url=queue_url,
        queue_name=queue_name,
    )
    dispatcher = WorkerDispatcher()
    timeout_seconds = max(int(poll_interval), 1)

    while True:
        task = queue.dequeue(timeout_seconds=timeout_seconds)
        if task is None:
            continue
        result = asyncio.run(dispatcher.dispatch(task))
        print(json.dumps(result.model_dump(), indent=2))


def _run_queue_consumers(args: argparse.Namespace) -> int:
    """Spawn and supervise queue consumer processes.

    Worker process lifecycle is tied to this parent process and supports
    graceful termination on keyboard interruption.
    """

    ctx = mp.get_context("spawn")
    processes: list = []

    for _ in range(args.workers):
        process = ctx.Process(
            target=_queue_consumer_loop,
            args=(
                args.queue_backend,
                args.queue_url,
                args.queue_name,
                args.poll_interval,
            ),
        )
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

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
    source: str,
    mode: str,
    task_file: str,
    workers: int,
    watch: bool,
    poll_interval: float,
    queue_backend: str,
    queue_url: str,
    queue_name: str,
):
    """Target function for running workers in a child process."""
    args = argparse.Namespace(
        mode=mode,
        task_file=task_file,
        source=source,
        workers=workers,
        watch=watch,
        poll_interval=poll_interval,
        queue_backend=queue_backend,
        queue_url=queue_url,
        queue_name=queue_name,
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
            args.source,
            args.mode,
            args.task_file,
            args.workers,
            args.watch,
            args.poll_interval,
            args.queue_backend,
            args.queue_url,
            args.queue_name,
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


def ingest_command(args: argparse.Namespace) -> int:
    """Run ingestion immediately for one file or in loop mode for a folder."""

    if args.file:
        _ingest_one_file(
            file_path=args.file,
            embedding_models=args.embedding_models,
            features=args.features,
            face_detector=args.face_detector,
            project_id=args.config_name,
            source="cli",
        )
        return 0

    seen_paths: set[str] = set()
    print("Starting folder ingest loop. Press Ctrl+C to stop.")
    while True:
        candidate_paths = _list_folder_files(args.folder)
        new_paths = [path for path in candidate_paths if path not in seen_paths]

        for path in new_paths:
            _ingest_one_file(
                file_path=path,
                embedding_models=args.embedding_models,
                features=args.features,
                face_detector=args.face_detector,
                project_id=args.config_name,
                source="cli-loop",
            )
            seen_paths.add(path)

        time.sleep(args.poll_interval)


def _build_ingest_task(
    path: str,
    embedding_models: list[str],
    features: list[str],
    face_detector: str,
    project_id: str | None,
    source: str,
):
    """Build one unit extraction task envelope from file path."""

    from faceapp_services.workers.contracts import (
        ExtractionUnitPayload,
        TaskEnvelope,
        TaskType,
    )

    payload = ExtractionUnitPayload(
        path=path,
        embedding_models=embedding_models,
        features=features,
        face_detector=face_detector,
        project_id=project_id,
    )

    return TaskEnvelope(
        id=str(ulid.ulid()),
        type=TaskType.EXTRACT_UNIT,
        payload=payload,
        source=source,
    )


def _dispatch_ingest_task_locally(task) -> dict:
    """Dispatch one task envelope directly through the local dispatcher."""
    try:
        from faceapp_services.workers.dispatcher import WorkerDispatcher
    except ImportError as exc:
        raise RuntimeError(
            "Ingest local dispatch dependencies are missing. "
            "Ensure libmagic is installed in the runtime environment."
        ) from exc

    dispatcher = WorkerDispatcher()
    result = asyncio.run(dispatcher.dispatch(task))
    return result.model_dump()


def _ingest_one_file(
    file_path: str,
    embedding_models: list[str],
    features: list[str],
    face_detector: str,
    project_id: str | None,
    source: str,
) -> None:
    """Build and dispatch one ingestion task immediately."""

    task = _build_ingest_task(
        path=file_path,
        embedding_models=embedding_models,
        features=features,
        face_detector=face_detector,
        project_id=project_id,
        source=source,
    )
    result = _dispatch_ingest_task_locally(task)
    print(json.dumps(result, indent=2))


def _list_folder_files(folder: str) -> list[str]:
    """List direct child files from folder in sorted order."""

    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []

    return [str(path) for path in sorted(folder_path.iterdir()) if path.is_file()]


def search_command(args: argparse.Namespace) -> int:
    """Search matches for one query file or loop over a query folder."""

    from faceapp.utils.processes.vector_index.qdrant import QdrantVectorStore
    from faceapp.utils.search import FaceSearch

    thresholds = _normalize_thresholds(
        thresholds=args.thresholds,
        num_models=len(args.embedding_models),
    )

    vector_store = QdrantVectorStore(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        path=os.getenv("QDRANT_PATH"),
    )
    finder = FaceSearch(
        vector_db=vector_store,
        embedding_models=args.embedding_models,
        model_thresholds=thresholds,
        project_id=args.config_name,
    )

    if args.file:
        _search_one_file(finder, args.file)
        return 0

    seen_paths: set[str] = set()
    print("Starting folder search loop. Press Ctrl+C to stop.")
    while True:
        candidate_paths = _list_folder_files(args.folder)
        new_paths = [path for path in candidate_paths if path not in seen_paths]

        for path in new_paths:
            _search_one_file(finder, path)
            seen_paths.add(path)

        time.sleep(args.poll_interval)


def _search_one_file(finder, file_path: str) -> None:
    """Run search for one query file and print results."""

    results = asyncio.run(finder.find(file_path))
    print(json.dumps({"query": file_path, "matches": results}, indent=2))


def _normalize_thresholds(
    thresholds: list[float],
    num_models: int,
) -> list[float]:
    """Match threshold list length to embedding model count.

    Accepts either one threshold shared by all models or one threshold per
    model. A default value is provided for single-model searches.
    """

    if num_models <= 1:
        return thresholds[:1] if thresholds else [0.4]

    if len(thresholds) == 1:
        return thresholds * num_models

    if len(thresholds) != num_models:
        raise ValueError(
            "--thresholds must contain one value " "or match --embedding-models count"
        )

    return thresholds


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    """Register shared server arguments on a command parser."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def _add_worker_args(parser: argparse.ArgumentParser) -> None:
    """Register shared worker arguments on a command parser."""
    parser.add_argument(
        "--source",
        choices=["queue", "task-file"],
        default="queue",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "multiprocess"],
        default="multiprocess",
    )
    parser.add_argument("--task-file", required=False)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--queue-backend",
        default=os.getenv("TASK_QUEUE_BACKEND", "redis"),
    )
    parser.add_argument(
        "--queue-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )
    parser.add_argument(
        "--queue-name",
        default=os.getenv("TASK_QUEUE_NAME", "faceapp:tasks"),
    )


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    """Register simplified ingestion arguments."""

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--file",
        help="Run ingestion immediately for one file.",
    )
    target_group.add_argument(
        "--folder",
        help="Run ingestion loop for folder files.",
    )
    parser.add_argument(
        "--embedding-models",
        action="append",
        default=["Facenet512"],
    )
    parser.add_argument(
        "--features",
        action="append",
        default=["age", "gender", "race", "emotion"],
    )
    parser.add_argument("--face-detector", default="mtcnn")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Config name/key from configs/projects.yaml.",
    )
    parser.add_argument(
        "--project-id",
        dest="config_name",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Folder loop polling interval in seconds.",
    )


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    """Register simplified search arguments."""

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--file",
        help="Search one query file immediately.",
    )
    target_group.add_argument(
        "--folder",
        help="Search loop for files in folder.",
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Config name/key from configs/projects.yaml.",
    )
    parser.add_argument(
        "--project-id",
        dest="config_name",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-models",
        action="append",
        default=["Facenet512"],
    )
    parser.add_argument(
        "--thresholds",
        action="append",
        type=float,
        default=[0.4],
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Folder loop polling interval in seconds.",
    )


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

    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest one file now or loop over folder",
    )
    _add_ingest_args(ingest)
    ingest.set_defaults(handler=ingest_command)

    search = subparsers.add_parser(
        "search",
        help="Search one file now or loop over folder",
    )
    _add_search_args(search)
    search.set_defaults(handler=search_command)

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

    run_ingest = run_subparsers.add_parser(
        "ingest",
        help="Ingest one file now or loop over folder",
    )
    _add_ingest_args(run_ingest)
    run_ingest.set_defaults(handler=ingest_command)

    run_search = run_subparsers.add_parser(
        "search",
        help="Search one file now or loop over folder",
    )
    _add_search_args(run_search)
    run_search.set_defaults(handler=search_command)

    return parser


def main() -> int:
    """CLI entry point for the `faceapp` command."""
    parser = _build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
