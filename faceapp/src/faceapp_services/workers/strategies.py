from __future__ import annotations

"""Strategy selection for worker extraction and indexing pipelines.

Strategies are resolved from project config and default to:
- extraction: local image extraction pipeline
- indexing: qdrant indexing pipeline
- Update this file whenever new pipelines are built
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

import faceapp.utils.pipelines as pipeline_registry
from faceapp._base.pipeline import Pipeline


class ExtractionStrategy(Protocol):
    """Creates extraction pipelines."""

    def build_pipeline(self) -> Pipeline: ...


class IndexingStrategy(Protocol):
    """Creates indexing pipelines."""

    def build_pipeline(self) -> Pipeline: ...


class LocalExtractionStrategy:
    """Default extraction strategy using local image processing."""

    def build_pipeline(self) -> Pipeline:
        return pipeline_registry.LocalImageExtractionPipeline()


class QdrantIndexingStrategy:
    """Default indexing strategy using Qdrant vector store pipeline."""

    def build_pipeline(self) -> Pipeline:
        return pipeline_registry.QdrantIndexingPipeline()


@dataclass
class ResolvedWorkerStrategies:
    """Resolved strategy instances and computed indexing config."""

    extraction: ExtractionStrategy
    indexing: IndexingStrategy
    index_config: dict[str, Any]


class WorkerStrategyResolver:
    """Resolves worker strategies from YAML config files."""

    def __init__(self, config_dir: str | None = None):
        self.config_dir = Path(
            config_dir or os.getenv("FACEAPP_CONFIG_DIR") or "./configs"
        )

    def resolve(
        self,
        project_id: str | None = None,
    ) -> ResolvedWorkerStrategies:
        """Resolve extraction/indexing strategies and index config."""

        project_config = self._load_project_config(project_id)

        extraction_name = self._resolve_extraction_name(project_config)
        indexing_name = self._resolve_indexing_name(project_config)
        index_config = self._resolve_index_config(project_config)

        return ResolvedWorkerStrategies(
            extraction=self._get_extraction_strategy(extraction_name),
            indexing=self._get_indexing_strategy(indexing_name),
            index_config=index_config,
        )

    def _load_project_config(self, project_id: str | None) -> dict[str, Any]:
        projects_cfg = self._load_yaml("projects.yaml")
        extraction_cfg = self._load_yaml("extraction.yaml")
        indexing_cfg = self._load_yaml("indexing.yaml")

        project_cfg = self._find_project_config(projects_cfg, project_id)
        if isinstance(project_cfg, dict):
            config_refs = project_cfg.get("config_refs", {})
        else:
            config_refs = {}

        extraction_profile = None
        indexing_profile = None
        vector_db = None
        if isinstance(config_refs, dict):
            extraction_profile = config_refs.get("extraction_profile")
            indexing_ref = config_refs.get("indexing", {})
            if isinstance(indexing_ref, dict):
                indexing_profile = indexing_ref.get("profile")
                vector_db = indexing_ref.get("vector_db")

        extraction_section = self._resolve_extraction_section(
            extraction_cfg,
            extraction_profile,
        )
        indexing_section = self._resolve_indexing_section(
            indexing_cfg,
            vector_db,
            indexing_profile,
        )

        merged = {}
        merged.update(extraction_section)
        merged.update(indexing_section)
        if isinstance(project_cfg, dict):
            merged.update(
                {
                    key: value
                    for key, value in project_cfg.items()
                    if key != "config_refs"
                }
            )
        return merged

    @staticmethod
    def _find_project_config(
        projects_cfg: dict[str, Any],
        project_id: str | None,
    ) -> dict[str, Any]:
        if not isinstance(projects_cfg, dict):
            return {}

        if project_id and project_id in projects_cfg:
            cfg = projects_cfg.get(project_id, {})
            return cfg if isinstance(cfg, dict) else {}

        if project_id:
            for cfg in projects_cfg.values():
                is_target = isinstance(cfg, dict)
                is_target = is_target and cfg.get("project_id") == project_id
                if is_target:
                    return cfg

        if projects_cfg:
            first = next(iter(projects_cfg.values()))
            if isinstance(first, dict):
                return first
        return {}

    def _resolve_extraction_section(
        self,
        extraction_cfg: dict[str, Any],
        profile_name: str | None,
    ) -> dict[str, Any]:
        profiles = extraction_cfg.get("profiles", {})
        defaults = extraction_cfg.get("defaults", {})
        selected_profile = profile_name or defaults.get("profile")
        if not selected_profile or not isinstance(profiles, dict):
            return {}
        section = profiles.get(selected_profile, {})
        return section if isinstance(section, dict) else {}

    def _resolve_indexing_section(
        self,
        indexing_cfg: dict[str, Any],
        vector_db: str | None,
        profile_name: str | None,
    ) -> dict[str, Any]:
        profiles = indexing_cfg.get("profiles", {})
        defaults = indexing_cfg.get("defaults", {})
        selected_db = vector_db or defaults.get("vector_db") or "qdrant"
        if not profile_name:
            by_db = defaults.get("profiles_by_vector_db", {})
            if isinstance(by_db, dict):
                profile_name = by_db.get(selected_db)

        section = {}
        if isinstance(profiles, dict) and profile_name:
            profile = profiles.get(profile_name, {})
            if isinstance(profile, dict):
                section = dict(profile)

        if "vector_db" not in section:
            section["vector_db"] = selected_db
        return section

    @staticmethod
    def _resolve_extraction_name(project_config: dict[str, Any]) -> str:
        extraction_cfg = project_config.get("extraction", {})
        if isinstance(extraction_cfg, dict):
            strategy = extraction_cfg.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                return strategy.strip().lower()
        return "local"

    @staticmethod
    def _resolve_indexing_name(project_config: dict[str, Any]) -> str:
        indexing_cfg = project_config.get("indexing", {})
        if isinstance(indexing_cfg, dict):
            strategy = indexing_cfg.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                return strategy.strip().lower()

        vector_db = project_config.get("vector_db")
        if isinstance(vector_db, str) and vector_db.strip():
            return vector_db.strip().lower()
        return "qdrant"

    @staticmethod
    def _resolve_index_config(
        project_config: dict[str, Any],
    ) -> dict[str, Any]:
        qdrant_cfg = project_config.get("qdrant_index_config")
        if isinstance(qdrant_cfg, dict):
            return qdrant_cfg
        chroma_cfg = project_config.get("index_config")
        if isinstance(chroma_cfg, dict):
            return chroma_cfg
        return {}

    @staticmethod
    def _get_extraction_strategy(name: str) -> ExtractionStrategy:
        extraction_strategies: dict[str, ExtractionStrategy] = {
            "local": LocalExtractionStrategy(),
        }
        return extraction_strategies.get(name, LocalExtractionStrategy())

    @staticmethod
    def _get_indexing_strategy(name: str) -> IndexingStrategy:
        indexing_strategies: dict[str, IndexingStrategy] = {
            "qdrant": QdrantIndexingStrategy(),
        }
        return indexing_strategies.get(name, QdrantIndexingStrategy())

    def _load_yaml(self, file_name: str) -> dict[str, Any]:
        path = self.config_dir / file_name
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
