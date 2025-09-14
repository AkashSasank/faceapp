from typing import Protocol


class Process(Protocol):

    def invoke(self, *args, **kwargs) -> dict:
        return NotImplemented

    async def ainvoke(self, *args, **kwargs) -> dict:
        return NotImplemented
