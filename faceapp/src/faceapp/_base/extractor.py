from abc import abstractmethod

from faceapp._base.base import Process, ProcessOutput


class Extractor(Process[ProcessOutput]):

    @abstractmethod
    async def extract(self, *args, **kwargs) -> ProcessOutput:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> ProcessOutput:
        return await self.extract(*args, **kwargs)
