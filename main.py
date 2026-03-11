"""
Main script for ingesting a folder of images and running a sample search
using the FaceApp programmatic interface.
"""

from __future__ import annotations

import json
import os

from faceapp.interface import FaceAppClient

# ---------------------------------------------------------------------------
# Constants – adjust these to point at the data you want to use
# ---------------------------------------------------------------------------

INGEST_FOLDER: str = "dataset/raw"  # folder whose images will be ingested
SEARCH_IMAGE: str = "dataset/faces/amm.png"  # query image for the sample search

PROJECT_ID: str = "multimodelproj001"  # matches the multi_embed_demo project
EMBEDDING_MODELS: list[str] = ["Facenet512", "VGG-Face", "DeepID"]
FEATURES: list[str] = ["age", "gender"]
FACE_DETECTOR: str = "mtcnn"
THRESHOLDS: list[float] = [0.97, 0.97, 0.97]  # one per model
TOP_K = None

QDRANT_URL: str = "http://localhost:6333"

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)

# ---------------------------------------------------------------------------


def ingest_folder(client: FaceAppClient, folder: str) -> None:
    """Iterate *folder* and ingest every image file found."""

    abs_folder = os.path.abspath(folder)
    image_files = [
        os.path.join(abs_folder, f)
        for f in sorted(os.listdir(abs_folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"[ingest] No image files found in '{folder}'")
        return

    print(f"[ingest] Found {len(image_files)} image(s) in '{folder}'")

    for i, file_path in enumerate(image_files, start=1):
        print(
            f"[ingest] ({i}/{len(image_files)}) {os.path.basename(file_path)} ...",
            end=" ",
            flush=True,
        )
        result = client.ingest(
            file_path=file_path,
            project_id=PROJECT_ID,
            embedding_models=EMBEDDING_MODELS,
            features=FEATURES,
            face_detector=FACE_DETECTOR,
            qdrant_url=QDRANT_URL,
        )
        status = result.get("status", "unknown")
        print(status)

    print("[ingest] Done.\n")


def run_search(client: FaceAppClient, query_image: str) -> None:
    """Run a similarity search for *query_image* and print the results."""

    abs_query = os.path.abspath(query_image)
    print(f"[search] Query image: {abs_query}")

    result = client.search(
        file_path=abs_query,
        project_id=PROJECT_ID,
        embedding_models=EMBEDDING_MODELS,
        thresholds=THRESHOLDS,
        top_k=TOP_K,
        qdrant_url=QDRANT_URL,
    )

    matches = result.get("matches", [])
    print(f"[search] {len(matches)} match(es) found:")
    print(json.dumps(matches, indent=2, default=str))


def main() -> None:
    client = FaceAppClient()
    ingest_folder(client, INGEST_FOLDER)
    run_search(client, SEARCH_IMAGE)


if __name__ == "__main__":
    main()
