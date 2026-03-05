"""Sphinx configuration for FaceApp documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

project = "faceapp"
author = "FaceApp"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep docs generation lightweight in local/dev CI by mocking heavy runtime deps.
autodoc_mock_imports = [
    "azure",
    "azure.search",
    "azure.search.documents",
    "boto3",
    "chromadb",
    "cv2",
    "deepface",
    "fastapi",
    "magic",
    "numpy",
    "pandas",
    "qdrant_client",
    "redis",
    "tensorflow",
    "tf_keras",
    "torch",
    "ulid",
    "uvicorn",
    "yaml",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = os.getenv("SPHINX_THEME", "alabaster")
html_static_path = ["_static"]
