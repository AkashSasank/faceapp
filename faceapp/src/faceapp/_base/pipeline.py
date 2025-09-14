from typing import AsyncGenerator, Dict

from faceapp._base.base import Process


class Pipeline(Process):

    def __init__(self, processes: Dict[str, Process]):
        self.processes = processes

    async def ainvoke(self, *args, **kwargs) -> [dict, AsyncGenerator[dict, None]]:
        output = {}
        for process_name, process in self.processes.items():
            updated_kwargs = kwargs | output
            output = await process.ainvoke(*args, **updated_kwargs)
        return output
