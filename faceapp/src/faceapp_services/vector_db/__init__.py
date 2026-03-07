from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from . import qdrant


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return payload if isinstance(payload, dict) else {}


def resolve_vector_db_name(config_name: str | None) -> str:
    config_dir = Path(os.getenv("FACEAPP_CONFIG_DIR") or "./configs")
    projects_cfg = _load_yaml(config_dir / "projects.yaml")
    indexing_cfg = _load_yaml(config_dir / "indexing.yaml")

    if not config_name:
        defaults = (
            indexing_cfg.get("defaults", {}) if isinstance(indexing_cfg, dict) else {}
        )
        if isinstance(defaults, dict):
            configured = defaults.get("vector_db")
            if isinstance(configured, str) and configured.strip():
                return configured.strip().lower()
        return "qdrant"

    project_cfg = (
        projects_cfg.get(config_name, {}) if isinstance(projects_cfg, dict) else {}
    )
    if isinstance(project_cfg, dict):
        config_refs = project_cfg.get("config_refs", {})
        if isinstance(config_refs, dict):
            indexing_ref = config_refs.get("indexing", {})
            if isinstance(indexing_ref, dict):
                configured = indexing_ref.get("vector_db")
                if isinstance(configured, str) and configured.strip():
                    return configured.strip().lower()

    defaults = (
        indexing_cfg.get("defaults", {}) if isinstance(indexing_cfg, dict) else {}
    )
    if isinstance(defaults, dict):
        configured = defaults.get("vector_db")
        if isinstance(configured, str) and configured.strip():
            return configured.strip().lower()
    return "qdrant"


def apply_runtime_connection(config_name: str | None) -> None:
    vector_db = resolve_vector_db_name(config_name)
    if vector_db == "qdrant":
        qdrant.apply_runtime_connection(config_name)


def supports_parallel_ingest(config_name: str | None) -> bool:
    vector_db = resolve_vector_db_name(config_name)
    if vector_db == "qdrant":
        return qdrant.is_server_mode(config_name)
    return False


def should_skip_file_before_embedding(
    file_path: str,
    project_id: str | None,
    embedding_models: list[str],
    config_name: str | None = None,
) -> bool:
    vector_db = resolve_vector_db_name(config_name)
    if vector_db == "qdrant":
        return qdrant.should_skip_file_before_embedding(
            file_path=file_path,
            project_id=project_id,
            embedding_models=embedding_models,
            config_name=config_name,
        )
    return False


def models_requiring_embedding(
    file_path: str,
    project_id: str | None,
    embedding_models: list[str],
    config_name: str | None = None,
) -> list[str]:
    vector_db = resolve_vector_db_name(config_name)
    if vector_db == "qdrant":
        return qdrant.models_requiring_embedding(
            file_path=file_path,
            project_id=project_id,
            embedding_models=embedding_models,
            config_name=config_name,
        )
    return list(embedding_models)


def build_vector_store(config_name: str | None):
    vector_db = resolve_vector_db_name(config_name)
    if vector_db == "qdrant":
        return qdrant.build_vector_store(config_name)

    raise ValueError(f"Unsupported vector DB for search: {vector_db}")
