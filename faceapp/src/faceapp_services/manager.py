from __future__ import annotations

"""Management commands for local development.

Usage examples:
- Ingest one file:
    - faceapp ingest --file ./dataset/raw/sample.jpg --config-name foo
    - faceapp run ingest --file ./dataset/raw/sample.jpg --config-name foo

- Ingest folder loop (single worker):
    - faceapp ingest --folder ./dataset/raw --config-name foo --poll-interval 2

- Ingest folder loop (concurrent workers for faster ingestion):
    - faceapp ingest --folder ./dataset/raw --config-name foo \
        --ingest-workers 4
    - faceapp run ingest --folder ./dataset/raw --config-name foo \
        --ingest-workers 4 \
        --embedding-models Facenet512 --features age --features gender

- Search one file:
    - faceapp search --file ./dataset/raw/query.jpg --config-name foo
    - faceapp run search --file ./dataset/raw/query.jpg --config-name foo

- Search folder loop:
    - faceapp search --folder ./dataset/test --config-name foo \
        --poll-interval 2
    - faceapp run search --folder ./dataset/test --config-name foo \
        --embedding-models Facenet512 --thresholds 0.4
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import ulid
import yaml
from faceapp_services.vector_db import (
    apply_runtime_connection,
    build_vector_store,
    models_requiring_embedding,
    supports_parallel_ingest,
)


def _json_default(value: Any) -> Any:
    """Normalize non-JSON-native values for CLI output serialization."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_pretty_json(payload: Any) -> str:
    """Render JSON for CLI prints, including numpy scalar/list support."""
    return json.dumps(payload, indent=2, default=_json_default)


def _print_ingest_debug_summary(result_payload: dict[str, Any]) -> None:
    """Print compact per-model ingest diagnostics for one worker result."""

    if not isinstance(result_payload, dict):
        return

    data = result_payload.get("data", {})
    if not isinstance(data, dict):
        return

    extraction = data.get("extraction", {})
    indexing = data.get("indexing", {})

    extraction_rows: list[dict[str, Any]] = []
    if isinstance(extraction, dict):
        maybe_rows = extraction.get("extractions", [])
        if isinstance(maybe_rows, list):
            extraction_rows = [row for row in maybe_rows if isinstance(row, dict)]

    raw_faces_by_model: dict[str, int] = {}
    for row in extraction_rows:
        model_name = str(row.get("embedding_model") or "unknown")
        raw_faces_by_model[model_name] = raw_faces_by_model.get(model_name, 0) + 1

    indexed_docs_by_index: dict[str, int] = {}
    if isinstance(indexing, dict):
        maybe_documents = indexing.get("documents", {})
        if isinstance(maybe_documents, dict):
            for index_name, count in maybe_documents.items():
                if isinstance(index_name, str):
                    try:
                        indexed_docs_by_index[index_name] = int(count)
                    except (TypeError, ValueError):
                        continue

    indexed_docs_by_model: dict[str, int] = {}
    if raw_faces_by_model:
        for model_name in raw_faces_by_model:
            model_suffix = f"_{model_name.lower()}"
            indexed_docs_by_model[model_name] = sum(
                count
                for index_name, count in indexed_docs_by_index.items()
                if index_name.endswith(model_suffix)
            )

    summary = {
        "ingest_debug": {
            "raw_faces_by_model": raw_faces_by_model,
            "indexed_docs_by_index": indexed_docs_by_index,
            "indexed_docs_by_model": indexed_docs_by_model,
        }
    }
    print(_to_pretty_json(summary))


def ingest_command(args: argparse.Namespace) -> int:
    """Run ingestion immediately for one file or in loop mode for a folder."""

    apply_runtime_connection(args.config_name)

    embedding_models, features, face_detector = _resolve_ingest_runtime_settings(
        config_name=args.config_name,
        cli_embedding_models=args.embedding_models,
        cli_features=args.features,
        cli_face_detector=args.face_detector,
    )

    ingest_workers = max(int(args.ingest_workers), 1)
    supports_parallel = supports_parallel_ingest(args.config_name)
    if ingest_workers > 1 and not supports_parallel:
        print(
            "Parallel ingest is not enabled for the active vector DB/runtime. "
            "Falling back to single-thread ingest to avoid store conflicts."
        )
        ingest_workers = 1

    if args.file:
        _ingest_one_file(
            file_path=args.file,
            embedding_models=embedding_models,
            features=features,
            face_detector=face_detector,
            project_id=args.config_name,
            source="cli",
        )
        return 0

    seen_paths: set[str] = set()
    print("Starting folder ingest loop. Press Ctrl+C to stop.")
    while True:
        candidate_paths = _list_folder_files(args.folder)
        new_paths = [path for path in candidate_paths if path not in seen_paths]

        if not new_paths:
            time.sleep(args.poll_interval)
            continue

        if ingest_workers == 1:
            for path in new_paths:
                print(f"Ingesting {path}...")
                _ingest_one_file(
                    file_path=path,
                    embedding_models=embedding_models,
                    features=features,
                    face_detector=face_detector,
                    project_id=args.config_name,
                    source="cli-loop",
                )
                seen_paths.add(path)
        else:
            failed_paths = _ingest_many_files_threaded(
                file_paths=new_paths,
                embedding_models=embedding_models,
                features=features,
                face_detector=face_detector,
                project_id=args.config_name,
                source="cli-loop",
                max_workers=ingest_workers,
            )
            for path in new_paths:
                if path not in failed_paths:
                    seen_paths.add(path)

        time.sleep(args.poll_interval)


def _ingest_many_files_threaded(
    file_paths: list[str],
    embedding_models: list[str],
    features: list[str],
    face_detector: str,
    project_id: str | None,
    source: str,
    max_workers: int,
) -> set[str]:
    """Ingest many files concurrently via process workers.

    Returns file paths that failed ingestion.
    """
    from faceapp_services.workers import MultiprocessWorkerRunner

    failed_paths: set[str] = set()
    task_to_path: dict[str, str] = {}
    tasks = []

    for path in file_paths:
        pending_models = models_requiring_embedding(
            file_path=path,
            project_id=project_id,
            embedding_models=embedding_models,
            config_name=project_id,
        )
        if not pending_models:
            print(f"Skipping duplicate file (already indexed): {path}")
            continue

        skipped_models = [
            model for model in embedding_models if model not in pending_models
        ]
        if skipped_models:
            print(
                "Skipping duplicate models for " f"{path}: {', '.join(skipped_models)}"
            )

        task = _build_ingest_task(
            path=path,
            embedding_models=pending_models,
            features=features,
            face_detector=face_detector,
            project_id=project_id,
            source=source,
        )
        tasks.append(task)
        task_to_path[task.id] = path

    if not tasks:
        return failed_paths

    runner = MultiprocessWorkerRunner(num_workers=max_workers)
    try:
        results = runner.run(tasks)
    except Exception as exc:
        for path in task_to_path.values():
            failed_paths.add(path)
        print(f"Parallel ingest run failed: {exc}")
        return failed_paths

    for result in results:
        path = task_to_path.get(result.task_id)
        if path is None:
            continue

        if result.status == "error":
            failed_paths.add(path)
            error_message = result.error or "unknown error"
            print(f"Ingest failed for {path}: {error_message}")
            continue

        payload = result.model_dump()
        _print_ingest_debug_summary(payload)
        print(_to_pretty_json(payload))

    return failed_paths


def _build_ingest_task(
    path: str,
    embedding_models: list[str],
    features: list[str],
    face_detector: str,
    project_id: str | None,
    source: str,
    meta: dict[str, Any] | None = None,
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
        meta=meta or {},
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

    pending_models = models_requiring_embedding(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        config_name=project_id,
    )
    if not pending_models:
        print(f"Skipping duplicate file (already indexed): {file_path}")
        return

    skipped_models = [
        model for model in embedding_models if model not in pending_models
    ]
    if skipped_models:
        print(
            "Skipping duplicate models for " f"{file_path}: {', '.join(skipped_models)}"
        )

    task = _build_ingest_task(
        path=file_path,
        embedding_models=pending_models,
        features=features,
        face_detector=face_detector,
        project_id=project_id,
        source=source,
    )
    result = _dispatch_ingest_task_locally(task)
    _print_ingest_debug_summary(result)
    print(_to_pretty_json(result))


def _resolve_ingest_runtime_settings(
    config_name: str | None,
    cli_embedding_models: list[str] | None,
    cli_features: list[str] | None,
    cli_face_detector: str | None,
) -> tuple[list[str], list[str], str]:
    """Resolve ingestion settings from config with explicit CLI override support."""

    config_dir = Path(os.getenv("FACEAPP_CONFIG_DIR") or "./configs")
    projects_cfg = _load_yaml(config_dir / "projects.yaml")
    extraction_cfg = _load_yaml(config_dir / "extraction.yaml")

    project_cfg = {}
    if isinstance(projects_cfg, dict) and isinstance(config_name, str):
        entry = projects_cfg.get(config_name, {})
        if isinstance(entry, dict):
            project_cfg = entry

    config_refs = project_cfg.get("config_refs", {})
    if not isinstance(config_refs, dict):
        config_refs = {}

    extraction_profile = _resolve_extraction_profile_name(extraction_cfg, config_refs)
    extraction_section = _resolve_extraction_profile_section(
        extraction_cfg,
        extraction_profile,
    )

    configured_models = _extract_embedding_models({}, extraction_section)
    configured_features = _extract_features(extraction_section)
    configured_face_detector = _extract_face_detector(extraction_section)

    embedding_models = list(cli_embedding_models or configured_models or ["Facenet512"])
    features = list(
        cli_features or configured_features or ["age", "gender", "race", "emotion"]
    )
    face_detector = cli_face_detector or configured_face_detector or "mtcnn"
    return embedding_models, features, face_detector


def _list_folder_files(folder: str) -> list[str]:
    """List direct child files from folder in sorted order."""

    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []

    return [str(path) for path in sorted(folder_path.iterdir()) if path.is_file()]


def search_command(args: argparse.Namespace) -> int:
    """Search matches for one query file or loop over a query folder."""

    apply_runtime_connection(args.config_name)

    from faceapp.utils.search import FaceSearch

    embedding_models, thresholds, top_k = _resolve_search_runtime_settings(
        config_name=args.config_name,
        cli_embedding_models=args.embedding_models,
        cli_thresholds=args.thresholds,
        cli_top_k=args.top_k,
    )

    vector_store = build_vector_store(args.config_name)
    finder = FaceSearch(
        vector_db=vector_store,
        embedding_models=embedding_models,
        model_thresholds=thresholds,
        top_k=top_k,
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
    print(_to_pretty_json({"query": file_path, "matches": results}))


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


def _resolve_search_runtime_settings(
    config_name: str,
    cli_embedding_models: list[str] | None,
    cli_thresholds: list[float] | None,
    cli_top_k: int | None,
) -> tuple[list[str], list[float], int | None]:
    """Resolve search model and threshold settings from config and CLI."""

    config_dir = Path(os.getenv("FACEAPP_CONFIG_DIR") or "./configs")
    projects_cfg = _load_yaml(config_dir / "projects.yaml")
    extraction_cfg = _load_yaml(config_dir / "extraction.yaml")
    search_cfg = _load_yaml(config_dir / "search.yaml")

    project_cfg = {}
    if isinstance(projects_cfg, dict):
        entry = projects_cfg.get(config_name, {})
        if isinstance(entry, dict):
            project_cfg = entry

    config_refs = project_cfg.get("config_refs", {})
    if not isinstance(config_refs, dict):
        config_refs = {}

    search_profile = _resolve_search_profile_name(search_cfg, config_refs)
    search_section = _resolve_profile_section(search_cfg, search_profile)

    extraction_profile = None
    search_extraction_profile = search_section.get("extraction_profile")
    if isinstance(search_extraction_profile, str) and search_extraction_profile:
        extraction_profile = search_extraction_profile
    elif isinstance(config_refs.get("extraction_profile"), str):
        extraction_profile = config_refs.get("extraction_profile")

    extraction_section = _resolve_extraction_profile_section(
        extraction_cfg,
        extraction_profile,
    )

    configured_models = _extract_embedding_models(
        search_section,
        extraction_section,
    )
    embedding_models = list(cli_embedding_models or configured_models or ["Facenet512"])

    configured_thresholds = _extract_thresholds(search_section)
    selected_thresholds = list(cli_thresholds or configured_thresholds or [0.4])
    normalized_thresholds = _normalize_thresholds(
        thresholds=selected_thresholds,
        num_models=len(embedding_models),
    )
    configured_top_k = _extract_top_k(search_section)
    resolved_top_k = _normalize_top_k(cli_top_k, configured_top_k)
    return embedding_models, normalized_thresholds, resolved_top_k


def _resolve_extraction_profile_name(
    extraction_cfg: dict,
    config_refs: dict,
) -> str | None:
    extraction_profile = config_refs.get("extraction_profile")
    extraction_ref = config_refs.get("extraction")
    if not extraction_profile and isinstance(extraction_ref, dict):
        extraction_profile = extraction_ref.get("profile")

    defaults = (
        extraction_cfg.get("defaults", {}) if isinstance(extraction_cfg, dict) else {}
    )
    if not extraction_profile and isinstance(defaults, dict):
        extraction_profile = defaults.get("profile")

    if isinstance(extraction_profile, str) and extraction_profile:
        return extraction_profile
    return None


def _resolve_search_profile_name(
    search_cfg: dict,
    config_refs: dict,
) -> str | None:
    search_profile = config_refs.get("search_profile")
    search_ref = config_refs.get("search")
    if not search_profile and isinstance(search_ref, dict):
        search_profile = search_ref.get("profile")

    defaults = search_cfg.get("defaults", {}) if isinstance(search_cfg, dict) else {}
    if not search_profile and isinstance(defaults, dict):
        search_profile = defaults.get("profile")

    if isinstance(search_profile, str) and search_profile:
        return search_profile
    return None


def _resolve_profile_section(config: dict, profile_name: str | None) -> dict:
    if not isinstance(config, dict) or not profile_name:
        return {}

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}

    section = profiles.get(profile_name, {})
    return section if isinstance(section, dict) else {}


def _resolve_extraction_profile_section(
    extraction_cfg: dict,
    profile_name: str | None,
) -> dict:
    section = _resolve_profile_section(extraction_cfg, profile_name)
    extraction = section.get("extraction")
    if isinstance(extraction, dict):
        return extraction
    return {}


def _extract_embedding_models(
    search_section: dict,
    extraction_section: dict,
) -> list[str]:
    search_models = search_section.get("embedding_models")
    if isinstance(search_models, list) and search_models:
        return [str(model) for model in search_models]

    extraction_models = extraction_section.get("embedding_models")
    if isinstance(extraction_models, list) and extraction_models:
        return [str(model) for model in extraction_models]

    return []


def _extract_thresholds(search_section: dict) -> list[float]:
    search_thresholds = search_section.get("thresholds")
    if isinstance(search_thresholds, list) and search_thresholds:
        return [float(item) for item in search_thresholds]

    alt_thresholds = search_section.get("similarity_thresholds")
    if isinstance(alt_thresholds, list) and alt_thresholds:
        return [float(item) for item in alt_thresholds]

    return []


def _extract_top_k(search_section: dict) -> int | None:
    for key in ("top_k", "k", "limit"):
        value = search_section.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _normalize_top_k(
    cli_top_k: int | None,
    configured_top_k: int | None,
) -> int | None:
    candidate = cli_top_k
    if candidate is None:
        candidate = configured_top_k

    if candidate is None:
        return None
    if candidate <= 0:
        return None
    return candidate


def _extract_features(extraction_section: dict) -> list[str]:
    configured_features = extraction_section.get("features")
    if isinstance(configured_features, list) and configured_features:
        return [str(feature) for feature in configured_features]
    return []


def _extract_face_detector(extraction_section: dict) -> str | None:
    configured_detector = extraction_section.get("face_detector")
    if isinstance(configured_detector, str) and configured_detector:
        return configured_detector
    return None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return payload if isinstance(payload, dict) else {}


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
        default=None,
        help=(
            "Embedding model for ingestion. Can be repeated. "
            "Defaults to configured extraction profile."
        ),
    )
    parser.add_argument(
        "--features",
        action="append",
        default=None,
        help=(
            "Face attributes to enrich during ingestion. Can be repeated. "
            "Defaults to configured extraction profile."
        ),
    )
    parser.add_argument(
        "--face-detector",
        default=None,
        help="Face detector backend. Defaults to configured extraction profile.",
    )
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
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent file ingests for --folder mode. "
            "Value may be forced to 1 when the active vector DB "
            "uses local mode."
        ),
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
        default=None,
        help=(
            "Embedding model for search. Can be repeated. "
            "Defaults to configured search profile/extraction profile."
        ),
    )
    parser.add_argument(
        "--thresholds",
        action="append",
        type=float,
        default=None,
        help=(
            "Similarity threshold for search. Can be repeated per model. "
            "Defaults to configured search profile."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Maximum results per model/query-face search. "
            "By default, returns all results passing threshold. "
            "Use 0 to disable cap explicitly."
        ),
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
