from faceapp._base.base import Process


class ExtractionFormatter(Process):

    def create_metadata(self, extractions: list, project_id: str = None):
        return list(map(lambda x: self.__format(x, project_id), extractions))

    def __format(self, extraction: dict, project_id: str = None):
        if project_id:
            project_id = project_id.strip().replace(" ", "_")
            index_name = f"{project_id}_{extraction.get('embedding_model').lower()}"
        else:
            index_name = extraction.get("embedding_model").lower()
        meta = extraction.get("meta")
        embedding = extraction.get("embedding")
        content = extraction.get("blob_name")
        del extraction["meta"]
        del extraction["embedding"]
        return {
            "index_name": index_name,
            "blob_name": content,
            "embedding": embedding,
            "metadata": extraction | meta,
        }

    async def ainvoke(self, extractions: list, *args, **kwargs) -> dict:
        meta = self.create_metadata(extractions)
        return {"extractions": meta}


class ImageMetadataAggregator(Process):
    async def ainvoke(
        self, documents: list, image_metadata: dict, *args, **kwargs
    ) -> dict:
        return {"image_metadata": image_metadata | {"documents": documents}}
