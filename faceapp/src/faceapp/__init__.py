"""Core FaceApp package.

This package contains reusable base abstractions and local process utilities
for extraction, indexing, and search workflows.
"""

from faceapp.interface import FaceAppClient, ingest_file, search_file


def hello() -> str:
    """Return a minimal greeting used by package smoke checks."""

    return "Hello from faceapp!"


__all__ = [
    "FaceAppClient",
    "ingest_file",
    "search_file",
    "hello",
]
