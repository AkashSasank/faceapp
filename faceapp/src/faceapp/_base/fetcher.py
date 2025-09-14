from abc import abstractmethod

from faceapp._base.base import Process


class Fetcher(Process):

    @abstractmethod
    async def fetch(self, *args, **kwargs) -> dict:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> dict:
        return await self.fetch(*args, **kwargs)
