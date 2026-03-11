from __future__ import annotations

import os
import platform
import resource
from collections import defaultdict
from typing import Any


def _resolve_psutil():
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return None
    return psutil


def _get_peak_rss_bytes() -> int | None:
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if not isinstance(max_rss, (int, float)):
        return None

    if platform.system() == "Darwin":
        return int(max_rss)

    return int(max_rss * 1024)


def get_process_memory_snapshot() -> dict[str, float | int]:
    """Collect a lightweight per-process memory snapshot."""

    snapshot: dict[str, float | int] = {
        "pid": os.getpid(),
    }

    psutil = _resolve_psutil()
    if psutil is not None:
        try:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
            snapshot["rss_mb"] = round(rss_bytes / (1024 * 1024), 2)
        except Exception:
            pass

    peak_rss_bytes = _get_peak_rss_bytes()
    if peak_rss_bytes is not None:
        snapshot["peak_rss_mb"] = round(peak_rss_bytes / (1024 * 1024), 2)

    return snapshot


def compact_extraction_output(
    extraction_output: dict[str, Any],
) -> dict[str, Any]:
    """Strip large embedding payloads from extraction output."""

    if not isinstance(extraction_output, dict):
        return {}

    compact_output = {
        key: value for key, value in extraction_output.items() if key != "extractions"
    }
    extractions = extraction_output.get("extractions", [])
    if not isinstance(extractions, list):
        compact_output["extractions"] = []
        compact_output["summary"] = {
            "rows": 0,
            "rows_by_model": {},
            "embedding_dimensions_by_model": {},
            "embedding_payload_mb_removed": 0.0,
        }
        return compact_output

    rows_by_model: dict[str, int] = defaultdict(int)
    embedding_dimensions_by_model: dict[str, set[int]] = defaultdict(set)
    sanitized_extractions: list[dict[str, Any]] = []
    estimated_removed_bytes = 0

    for extraction in extractions:
        if not isinstance(extraction, dict):
            continue

        sanitized = dict(extraction)
        model_name = str(sanitized.get("embedding_model") or "unknown")
        rows_by_model[model_name] += 1

        embedding = sanitized.pop("embedding", None)
        if isinstance(embedding, (list, tuple)):
            embedding_dimensions_by_model[model_name].add(len(embedding))
            sanitized["embedding_dimensions"] = len(embedding)
            estimated_removed_bytes += len(embedding) * 8

        sanitized_extractions.append(sanitized)

    compact_output["extractions"] = sanitized_extractions
    compact_output["summary"] = {
        "rows": len(sanitized_extractions),
        "rows_by_model": dict(sorted(rows_by_model.items())),
        "embedding_dimensions_by_model": {
            model: sorted(dimensions)
            for model, dimensions in sorted(
                embedding_dimensions_by_model.items(),
            )
        },
        "embedding_payload_mb_removed": round(
            estimated_removed_bytes / (1024 * 1024),
            4,
        ),
    }
    return compact_output
