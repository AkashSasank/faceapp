import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from faceapp._base.base import Process, ProcessOutput
from faceapp.utils.processes.process_outputs import (
    ImageMetadataOutput,
    MetadataFormattingOutput,
)


class ExtractionFormatter(Process):

    def create_metadata(self, extractions: list, project_id: str = None):
        image_hash_cache: dict[str, str] = {}
        formatted_data = list(
            map(
                lambda x: self.__format(
                    x,
                    project_id,
                    image_hash_cache,
                ),
                extractions,
            )
        )
        num_embedding_models = len(
            list({i[1].get("embedding_model", 1) for i in formatted_data})
        )
        num_faces = len(formatted_data) // max(
            num_embedding_models, 1
        )  # To prevent division by zero
        # when no faces detected
        embeddings = []
        metadata = []
        for data in formatted_data:
            embeddings.append(data[0])
            meta = data[1]
            meta["num_faces"] = num_faces
            metadata.append(meta)
        return embeddings, metadata

    def __format(
        self,
        extraction: dict,
        project_id: str = None,
        image_hash_cache: Optional[Dict[str, str]] = None,
    ):
        if project_id:
            project_id = project_id.strip().replace(" ", "_")
            embedding_model = extraction.get("embedding_model").lower()
            index_name = f"{project_id}_{embedding_model}"
        else:
            index_name = extraction.get("embedding_model").lower()

        image_hash = self._compute_image_hash(
            blob_name=extraction.get("blob_name"),
            image_hash_cache=image_hash_cache,
        )
        dedupe_hash = self._compute_dedupe_hash(
            extraction=extraction,
            image_hash=image_hash,
        )

        face_keys = [
            "facial_area",
            "face_confidence",
            "age",
            "dominant_gender",
            "dominant_race",
            "dominant_emotion",
            "detector",
            "embedding_model",
            "blob_name",
        ]
        face = dict()
        for key in face_keys:
            val = extraction.get(key)
            if isinstance(val, dict):
                # Flatten the meta dict
                face = face | val
                if key == "facial_area":
                    left_eye = face.get("left_eye")
                    right_eye = face.get("right_eye")
                    if left_eye:
                        face["left_eye_x"] = str(left_eye[0])
                        face["left_eye_y"] = str(left_eye[1])
                        del face["left_eye"]
                    if right_eye:
                        face["right_eye_x"] = str(right_eye[0])
                        face["right_eye_y"] = str(right_eye[1])
                        del face["right_eye"]

                continue

            face[key] = extraction.get(key)
        embedding = extraction.get("embedding")
        return (
            embedding,
            {
                "created_on": datetime.now().isoformat(),
                "updated_on": datetime.now().isoformat(),
                "index_name": index_name,
                "image_hash": image_hash,
                "dedupe_hash": dedupe_hash,
            }
            | face,
        )

    @staticmethod
    def _compute_image_hash(
        blob_name: Any,
        image_hash_cache: Optional[Dict[str, str]] = None,
    ) -> str:
        blob_name_str = str(blob_name or "")
        if image_hash_cache and blob_name_str in image_hash_cache:
            return image_hash_cache[blob_name_str]

        path = Path(blob_name_str)
        hasher = hashlib.sha256()
        if blob_name_str and path.is_file():
            with path.open("rb") as file_obj:
                while True:
                    chunk = file_obj.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
        else:
            hasher.update(blob_name_str.encode("utf-8"))

        image_hash = hasher.hexdigest()
        if image_hash_cache is not None:
            image_hash_cache[blob_name_str] = image_hash
        return image_hash

    @staticmethod
    def _compute_dedupe_hash(extraction: dict, image_hash: str) -> str:
        metadata_fingerprint = {
            "embedding_model": extraction.get("embedding_model"),
            "detector": extraction.get("detector"),
            "facial_area": extraction.get("facial_area"),
        }
        metadata_hash = hashlib.sha256(
            json.dumps(
                metadata_fingerprint,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return hashlib.sha256(
            f"{image_hash}:{metadata_hash}".encode("utf-8")
        ).hexdigest()

    async def ainvoke(
        self, extractions: list, project_id: str, *args, **kwargs
    ) -> ProcessOutput:
        embeddings, metadata = self.create_metadata(extractions, project_id)
        return MetadataFormattingOutput(
            embeddings=embeddings,
            metadata=metadata,
        )


class ImageMetadataAggregator(Process):
    async def ainvoke(
        self, documents: list, image_metadata: dict, *args, **kwargs
    ) -> ProcessOutput:
        return ImageMetadataOutput(
            image_metadata=image_metadata | {"documents": documents}
        )
