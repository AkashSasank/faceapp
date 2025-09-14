from abc import abstractmethod

from faceapp._base.base import Process


class Extractor(Process):

    @abstractmethod
    async def extract(self, *args, **kwargs) -> dict:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> dict:
        return await self.extract(*args, **kwargs)
