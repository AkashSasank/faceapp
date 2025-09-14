from faceapp._base.base import Process


class LocalImageExtractionFormatter(Process):

    def create_metadata(self, extractions: list):
        return list(map(lambda x: self.__format(x), extractions))

    def __format(self, extraction: dict):
        index_name = extraction.get("embedding_model").lower()
        meta = extraction.get("meta")
        embedding = extraction.get("embedding")
        content = extraction.get("file_ref")
        del extraction["meta"]
        del extraction["embedding"]
        return {
            "index_name": index_name,
            "content": content,
            "embedding": embedding,
            "metadata": extraction | meta,
        }

    async def ainvoke(self, extractions: list, *args, **kwargs) -> dict:
        meta = self.create_metadata(extractions)
        return {"extractions": meta}
