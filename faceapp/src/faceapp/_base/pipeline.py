from typing import AsyncGenerator, Dict

from faceapp._base.base import Process


class Pipeline(Process):

    def __init__(self, processes: Dict[str, Process], name: str):
        self.processes = processes
        self.name = name

    async def ainvoke(self, *args, **kwargs) -> [dict, AsyncGenerator[dict, None]]:
        output = {}
        updated_kwargs = kwargs | output
        process_names = list(self.processes.keys())
        print("Pipeline name: ", self.name)
        print("List of sub-processes: ", process_names)
        for process_name, process in self.processes.items():
            print("Current process: ", process_name)
            output = await process.ainvoke(*args, **updated_kwargs)
            updated_kwargs = updated_kwargs | output
            print("Process ", process_name, ": Completed")
        return output
