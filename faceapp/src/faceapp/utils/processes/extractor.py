import asyncio
from itertools import product
from typing import List

from deepface import DeepFace

from faceapp._base.extractor import Extractor


class FaceAnalyser:

    @staticmethod
    def analyse(path, features: [List, None], face_detector: str, align_face=True):
        faces = DeepFace.analyze(
            img_path=path,
            actions=features,
            align=align_face,
            detector_backend=face_detector,
            silent=True,
        )
        return {"faces": faces, "features": features}


class FaceEmbedder:

    @staticmethod
    def represent_faces(
        path, face_detector: str, embedding_model: str, align_face=True
    ):
        embedding_objs = DeepFace.represent(
            img_path=path,
            detector_backend=face_detector,
            align=align_face,
            model_name=embedding_model,
        )
        return {
            "embedding_objs": embedding_objs,
            "embedding_model": embedding_model,
        }


class FaceExtractor(Extractor):
    analyser = FaceAnalyser()
    embedder = FaceEmbedder()

    __features = ["age", "gender", "race", "emotion"]
    __detectors = [
        "opencv",
        "ssd",
        "dlib",
        "mtcnn",
        "fastmtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yolov11s",
        "yolov11n",
        "yolov11m",
        "yunet",
        "centerface",
    ]

    __models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
        "GhostFaceNet",
        "Buffalo_L",
    ]

    def __validate_detector(self, detector: str):
        assert detector in self.__detectors

    def __validate_model(self, model: list):
        assert len(set(model + self.__models)) == len(self.__models)

    def __validate_features(self, features: list):
        if features:
            assert len(set(features + self.__features)) == len(self.__features)
            return features
        else:
            return self.__features

    async def extract(
        self,
        path,
        meta: dict,
        embedding_models: list,
        features: list,
        face_detector: str = "mtcnn",
        *args,
        **kwargs,
    ) -> dict:
        features = self.__validate_features(features)
        self.__validate_model(embedding_models)
        self.__validate_detector(face_detector)
        threads = list()
        threads.extend(
            [
                asyncio.to_thread(
                    self.embedder.represent_faces,
                    path=path,
                    face_detector=face_detector,
                    align_face=False,
                    embedding_model=model,
                )
                for model in embedding_models
            ]
        )
        threads.append(
            asyncio.to_thread(
                self.analyser.analyse,
                path=path,
                face_detector=face_detector,
                align_face=False,
                features=features,
            )
        )
        res = await asyncio.gather(*threads)
        results = {
            "embeddings": res[0 : len(embedding_models)],
            "analysis": res[len(embedding_models) :],
            "detector": face_detector,
            "blob_name": path,
            "meta": meta,
        }
        results = self.clean_extractions(results)
        return results

    def clean_extractions(
        self,
        extractions: dict,
        face_threshold: float = 0.9,
        iou_threshold: float = 0.95,
    ) -> dict:
        blob_name = extractions["blob_name"]
        detector = extractions["detector"]
        meta = extractions["meta"]
        analysis = extractions["analysis"][0]
        analysis_data, analysis_bb = self.__clean_analysis_data(
            analysis, face_threshold
        )
        embeddings = extractions["embeddings"]
        embedding_data, embedding_bb = self.__clean_embedding_data(
            embeddings, face_threshold
        )

        iou = list(
            map(self.__intersection_over_union, product(embedding_bb, analysis_bb))
        )
        indexes = list(
            product(
                [i for i in range(len(embedding_bb))],
                [j for j in range(len(analysis_bb))],
            )
        )
        filtered_indexes = list(
            map(
                lambda x: x[1],
                list(
                    filter(
                        lambda i: i[1] if i[0] >= iou_threshold else None,
                        zip(iou, indexes),
                    )
                ),
            )
        )
        final_data = list()
        for index in filtered_indexes:
            if index:
                d = embedding_data[index[0]] | analysis_data[index[1]]
                del d["region"]
                d["detector"] = detector
                d["blob_name"] = blob_name
                d["meta"] = meta
                final_data.append(d)
        return {
            "extractions": final_data,
            "image_metadata": {
                "num_faces": len(analysis_data),
            },
        }

    @staticmethod
    def __clean_analysis_data(analysis, face_threshold):
        analysis_data = []
        analysis_bb = []
        faces = analysis["faces"]
        features = analysis["features"]
        for face in faces:
            if face["face_confidence"] > face_threshold:
                for f in features:
                    if f == "age":
                        continue
                    del face[f]
                analysis_bb.append(list(face["region"].values())[0:4])
                analysis_data.append(face)
        return analysis_data, analysis_bb

    @staticmethod
    def __clean_embedding_data(embeddings, face_threshold):
        embedding_data = []
        embedding_bb = []
        for embedding in embeddings:
            objects = embedding["embedding_objs"]
            embedding_model = embedding["embedding_model"]
            for obj in objects:
                if obj["face_confidence"] > face_threshold:
                    embedding_bb.append(list(obj["facial_area"].values())[0:4])
                    obj["embedding_model"] = embedding_model
                    embedding_data.append(obj)
        return embedding_data, embedding_bb

    @staticmethod
    def __intersection_over_union(boxes):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Parameters:
            box1, box2 = boxes
            box1: list or tuple [x, y, w, h]
            box2: list or tuple [x, y, w, h]
                (x, y) = top-left corner, w = width, h = height

        Returns:
            float: IoU value between 0 and 1
        """
        box1, box2 = boxes
        # Convert boxes to (x1, y1, x2, y2)
        x1_min, y1_min, w1, h1 = box1
        x1_max, y1_max = x1_min + w1, y1_min + h1

        x2_min, y2_min, w2, h2 = box2
        x2_max, y2_max = x2_min + w2, y2_min + h2

        # Intersection rectangle
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        intersection = inter_w * inter_h

        # Areas
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        # IoU
        if union == 0:
            return 0.0
        return intersection / union
