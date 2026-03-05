from __future__ import annotations

"""Typed contracts for worker tasks and results.

These models define the boundary between task producers and worker handlers.
"""

from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported worker task kinds."""

    EXTRACT_UNIT = "extract.unit"
    EXTRACT_BATCH = "extract.batch"


class ExtractionBasePayload(BaseModel):
    """Common extraction settings shared by unit and batch tasks."""

    embedding_models: List[str]
    features: List[str]
    face_detector: str = "mtcnn"
    project_id: Optional[str] = None


class ExtractionUnitPayload(ExtractionBasePayload):
    """Payload for extracting faces from one image path."""

    path: str


class ExtractionBatchPayload(ExtractionBasePayload):
    """Payload for extracting faces from multiple image paths."""

    paths: List[str] = Field(default_factory=list)


TaskPayload = Union[ExtractionUnitPayload, ExtractionBatchPayload]


class TaskEnvelope(BaseModel):
    """Task wrapper consumed by the worker dispatcher."""

    id: str
    type: TaskType
    payload: TaskPayload
    source: Optional[str] = None


class WorkerResult(BaseModel):
    """Standardized worker response for success and error outcomes."""

    task_id: str
    task_type: TaskType
    status: Literal["success", "error"]
    data: Any = None
    error: Optional[str] = None
