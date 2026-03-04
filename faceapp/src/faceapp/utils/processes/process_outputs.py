from typing import Any, Optional

from pydantic import Field

from faceapp._base.base import ProcessOutput


class FetchOutput(ProcessOutput):
    path: str
    meta: dict[str, Any] = Field(default_factory=dict)


class FaceExtractionOutput(ProcessOutput):
    extractions: list[dict[str, Any]] = Field(default_factory=list)


class MetadataFormattingOutput(ProcessOutput):
    embeddings: list[Any] = Field(default_factory=list)
    metadata: list[dict[str, Any]] = Field(default_factory=list)


class ImageMetadataOutput(ProcessOutput):
    image_metadata: dict[str, Any] = Field(default_factory=dict)


class CleanupOutput(ProcessOutput):
    cleanup_status: bool = Field(default=False)


class ChromaLoadOutput(ProcessOutput):
    documents: Any = Field(default=None)


class AzureIndexLoadOutput(ProcessOutput):
    documents: list[dict[str, Any]] = Field(default_factory=list)
    blob_name: Optional[str] = None
