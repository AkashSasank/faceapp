from datetime import datetime

import ulid

from faceapp._base.base import Process


class ExtractionFormatter(Process):

    def create_metadata(self, extractions: list, project_id: str = None):
        formatted_data = list(map(lambda x: self.__format(x, project_id), extractions))
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

    def __format(self, extraction: dict, project_id: str = None):
        if project_id:
            project_id = project_id.strip().replace(" ", "_")
            index_name = f"{project_id}_{extraction.get('embedding_model').lower()}"
        else:
            index_name = extraction.get("embedding_model").lower()
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
            }
            | face,
        )

    async def ainvoke(
        self, extractions: list, project_id: str, *args, **kwargs
    ) -> dict:
        embeddings, metadata = self.create_metadata(extractions, project_id)
        return {
            "embeddings": embeddings,
            "metadata": metadata,
        }


class ImageMetadataAggregator(Process):
    async def ainvoke(
        self, documents: list, image_metadata: dict, *args, **kwargs
    ) -> dict:
        return {"image_metadata": image_metadata | {"documents": documents}}
