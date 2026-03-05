import yaml


def load_config(config_path, config_name: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config.get(config_name, config)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def _resolve_extraction_config(
    extraction_config: dict, extraction_profile: str | None
) -> dict:
    profiles = extraction_config.get("profiles", {})
    default_profile = extraction_config.get("defaults", {}).get("profile")
    profile_name = extraction_profile or default_profile
    if not profile_name:
        return {}
    return profiles.get(profile_name, {})


def _resolve_indexing_config(
    indexing_config: dict,
    vector_db: str | None,
    indexing_profile: str | None,
) -> dict:
    profiles = indexing_config.get("profiles", {})
    defaults = indexing_config.get("defaults", {})
    default_vector_db = defaults.get("vector_db")
    selected_vector_db = vector_db or default_vector_db

    profiles_by_db = defaults.get("profiles_by_vector_db", {})
    profile_name = indexing_profile or profiles_by_db.get(selected_vector_db)
    if not profile_name:
        return {}

    profile = profiles.get(profile_name, {})
    if not isinstance(profile, dict):
        return {}

    resolved = dict(profile)
    if selected_vector_db and "vector_db" not in resolved:
        resolved["vector_db"] = selected_vector_db
    return resolved


def load_project_config(config_dir: str, project_name: str) -> dict:
    projects_cfg = _load_yaml(f"{config_dir}/projects.yaml")
    extraction_cfg = _load_yaml(f"{config_dir}/extraction.yaml")
    indexing_cfg = _load_yaml(f"{config_dir}/indexing.yaml")

    project_cfg = projects_cfg.get(project_name, {})
    if not isinstance(project_cfg, dict):
        return {}

    config_refs = project_cfg.get("config_refs", {})
    extraction_profile = None
    vector_db = None
    indexing_profile = None

    if isinstance(config_refs, dict):
        extraction_profile = config_refs.get("extraction_profile")
        indexing_ref = config_refs.get("indexing", {})
        if isinstance(indexing_ref, dict):
            vector_db = indexing_ref.get("vector_db")
            indexing_profile = indexing_ref.get("profile")

    extraction_resolved = _resolve_extraction_config(
        extraction_cfg,
        extraction_profile,
    )
    indexing_resolved = _resolve_indexing_config(
        indexing_cfg,
        vector_db,
        indexing_profile,
    )

    merged = dict(project_cfg)
    merged.pop("config_refs", None)
    merged = _deep_merge(merged, extraction_resolved)
    merged = _deep_merge(merged, indexing_resolved)
    return merged
