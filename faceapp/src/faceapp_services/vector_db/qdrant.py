from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from faceapp.utils.processes.vector_index.qdrant import QdrantVectorStore
from faceapp.utils.processes.vector_index.qdrant import (
    models_requiring_embedding as qdrant_models_requiring_embedding,
)
from faceapp.utils.processes.vector_index.qdrant import (
    should_skip_file_before_embedding as qdrant_should_skip_before_embedding,
)


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _first_non_empty_str(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return payload if isinstance(payload, dict) else {}


def _project_qdrant_overrides(
    config_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not config_name:
        return None, None, None

    config_dir = Path(os.getenv("FACEAPP_CONFIG_DIR") or "./configs")
    projects_cfg = _load_yaml(config_dir / "projects.yaml")
    if not isinstance(projects_cfg, dict):
        return None, None, None

    project_cfg = projects_cfg.get(config_name)
    if not isinstance(project_cfg, dict):
        return None, None, None

    config_refs = project_cfg.get("config_refs", {})
    indexing_cfg = config_refs.get("indexing", {})
    qdrant_cfg = {}
    if isinstance(indexing_cfg, dict):
        qdrant_ref = indexing_cfg.get("qdrant")
        if isinstance(qdrant_ref, dict):
            qdrant_cfg = qdrant_ref

    url = _first_non_empty_str(
        qdrant_cfg.get("url"),
        (indexing_cfg.get("qdrant_url") if isinstance(indexing_cfg, dict) else None),
        project_cfg.get("qdrant_url"),
    )
    api_key = _first_non_empty_str(
        qdrant_cfg.get("api_key"),
        (
            indexing_cfg.get("qdrant_api_key")
            if isinstance(indexing_cfg, dict)
            else None
        ),
        project_cfg.get("qdrant_api_key"),
    )
    path = _first_non_empty_str(
        qdrant_cfg.get("path"),
        (indexing_cfg.get("qdrant_path") if isinstance(indexing_cfg, dict) else None),
        project_cfg.get("qdrant_path"),
    )
    return url, api_key, path


def _project_env_variables(
    config_name: str | None,
) -> dict[str, str]:
    if not config_name:
        return {}

    config_dir = Path(os.getenv("FACEAPP_CONFIG_DIR") or "./configs")
    projects_cfg = _load_yaml(config_dir / "projects.yaml")
    env_cfg = _load_yaml(config_dir / "environments.yaml")

    project_cfg = (
        projects_cfg.get(config_name, {}) if isinstance(projects_cfg, dict) else {}
    )
    if not isinstance(project_cfg, dict):
        return {}

    config_refs = project_cfg.get("config_refs", {})
    if not isinstance(config_refs, dict):
        config_refs = {}

    env_key = config_refs.get("env_config_key")
    env_ref = config_refs.get("env")
    if not env_key and isinstance(env_ref, dict):
        env_key = env_ref.get("key")
    if not isinstance(env_key, str) or not env_key.strip():
        env_key = config_name

    defaults_section = env_cfg.get("defaults", {}) if isinstance(env_cfg, dict) else {}
    defaults_vars = {}
    if isinstance(defaults_section, dict):
        maybe_vars = defaults_section.get("variables", {})
        if isinstance(maybe_vars, dict):
            defaults_vars = maybe_vars

    env_section = env_cfg.get(env_key, {}) if isinstance(env_cfg, dict) else {}
    section_vars = {}
    if isinstance(env_section, dict):
        maybe_vars = env_section.get("variables", {})
        if isinstance(maybe_vars, dict):
            section_vars = maybe_vars

    variables = {**defaults_vars, **section_vars}
    if not isinstance(variables, dict):
        return {}

    resolved: dict[str, str] = {}
    for key, value in variables.items():
        if not isinstance(key, str):
            continue
        cleaned = _first_non_empty_str(value)
        if cleaned is not None:
            resolved[key] = cleaned
    return resolved


def resolve_connection(
    config_name: str | None = None,
) -> tuple[str | None, str | None, str | None, bool]:
    env_url = _clean_env("QDRANT_URL")
    env_api_key = _clean_env("QDRANT_API_KEY")
    env_path = _clean_env("QDRANT_PATH")
    profile_vars = _project_env_variables(config_name)
    profile_url = _first_non_empty_str(profile_vars.get("QDRANT_URL"))
    profile_api_key = _first_non_empty_str(profile_vars.get("QDRANT_API_KEY"))
    profile_path = _first_non_empty_str(profile_vars.get("QDRANT_PATH"))
    cfg_url, cfg_api_key, cfg_path = _project_qdrant_overrides(config_name)

    qdrant_url = env_url or profile_url or cfg_url
    qdrant_api_key = env_api_key or profile_api_key or cfg_api_key
    qdrant_path = env_path or profile_path or cfg_path

    if qdrant_url:
        return qdrant_url, qdrant_api_key, None, True

    return None, qdrant_api_key, qdrant_path, False


def apply_runtime_connection(config_name: str | None) -> None:
    qdrant_url, qdrant_api_key, qdrant_path, _ = resolve_connection(
        config_name=config_name,
    )

    if qdrant_url:
        os.environ["QDRANT_URL"] = qdrant_url
        os.environ.pop("QDRANT_PATH", None)
    elif qdrant_path:
        os.environ["QDRANT_PATH"] = qdrant_path
        os.environ.pop("QDRANT_URL", None)

    if qdrant_api_key:
        os.environ["QDRANT_API_KEY"] = qdrant_api_key


def is_server_mode(config_name: str | None = None) -> bool:
    _, _, _, server_mode = resolve_connection(config_name)
    return server_mode


def should_skip_file_before_embedding(
    file_path: str,
    project_id: str | None,
    embedding_models: list[str],
    config_name: str | None = None,
) -> bool:
    qdrant_url, qdrant_api_key, qdrant_path, _ = resolve_connection(
        config_name,
    )
    return qdrant_should_skip_before_embedding(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        url=qdrant_url,
        api_key=qdrant_api_key,
        path=qdrant_path,
    )


def models_requiring_embedding(
    file_path: str,
    project_id: str | None,
    embedding_models: list[str],
    config_name: str | None = None,
) -> list[str]:
    qdrant_url, qdrant_api_key, qdrant_path, _ = resolve_connection(
        config_name,
    )
    return qdrant_models_requiring_embedding(
        file_path=file_path,
        project_id=project_id,
        embedding_models=embedding_models,
        url=qdrant_url,
        api_key=qdrant_api_key,
        path=qdrant_path,
    )


def build_vector_store(config_name: str | None):
    qdrant_url, qdrant_api_key, qdrant_path, _ = resolve_connection(
        config_name,
    )
    return QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        path=qdrant_path,
    )
