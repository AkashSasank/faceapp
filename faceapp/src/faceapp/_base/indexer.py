from abc import abstractmethod

from faceapp._base.base import Process, ProcessOutput


class Indexer(Process[ProcessOutput]):
    @abstractmethod
    async def load(self, *args, **kwargs) -> ProcessOutput:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> ProcessOutput:
        return await self.load(*args, **kwargs)

    async def search(self, *args, **kwargs) -> list | dict:
        return NotImplemented
