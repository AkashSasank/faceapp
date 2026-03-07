from abc import abstractmethod
from typing import Union

from faceapp._base.base import Process, ProcessOutput


class Indexer(Process[ProcessOutput]):
    @abstractmethod
    async def load(self, *args, **kwargs) -> ProcessOutput:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> ProcessOutput:
        return await self.load(*args, **kwargs)

    async def search(self, *args, **kwargs) -> Union[list, dict]:
        return NotImplemented

    def existing_dedupe_hashes(
        self,
        index_name: str,
        dedupe_hashes: set[str],
    ) -> set[str]:
        """Optional capability hook for dedupe decorators.

        Backends that support efficient hash lookups can override this method.
        Default implementation returns empty set
        (no pre-existing hashes known).
        """
        return set()
