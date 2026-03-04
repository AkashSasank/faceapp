from abc import abstractmethod

from faceapp._base.base import Process, ProcessOutput


class Fetcher(Process[ProcessOutput]):

    @abstractmethod
    async def fetch(self, *args, **kwargs) -> ProcessOutput:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> ProcessOutput:
        return await self.fetch(*args, **kwargs)
