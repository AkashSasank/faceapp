from abc import abstractmethod

from faceapp._base.base import Process


class Indexer(Process):

    @abstractmethod
    async def load(self, *args, **kwargs) -> dict:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> dict:
        return await self.load(*args, **kwargs)

    def search(self, *args, **kwargs) -> [list, dict]:
        return NotImplemented
