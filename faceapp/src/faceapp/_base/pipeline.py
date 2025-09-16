from typing import AsyncGenerator, Dict

from faceapp._base.base import Process


class Pipeline(Process):

    def __init__(self, processes: Dict[str, Process]):
        self.processes = processes

    async def ainvoke(self, *args, **kwargs) -> [dict, AsyncGenerator[dict, None]]:
        output = {}
        updated_kwargs = kwargs | output
        for process_name, process in self.processes.items():
            print("#" * 100)
            print("Process name:", process_name)
            print("Process input items:", list(updated_kwargs.keys()))
            output = await process.ainvoke(*args, **updated_kwargs)
            print("Process output items:", list(output.keys()))
            updated_kwargs = updated_kwargs | output
        return output
