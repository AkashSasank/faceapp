from typing import Any, AsyncGenerator, Dict, Union

from pydantic import BaseModel

from faceapp._base.base import Process, ProcessOutput


class Pipeline:
    def __init__(self, processes: Dict[str, Process], name: str):
        self.processes = processes
        self.name = name

    @staticmethod
    def _result_to_kwargs(result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, ProcessOutput):
            return result.to_dict()
        if isinstance(result, BaseModel):
            return result.model_dump(exclude_none=True)
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unsupported process output type: {type(result)!r}.")

    async def ainvoke(
        self, *args, **kwargs
    ) -> Union[ProcessOutput, AsyncGenerator[ProcessOutput, None]]:
        output = ProcessOutput.model_validate({})
        updated_kwargs = dict(kwargs)
        process_names = list(self.processes.keys())
        print("Pipeline name: ", self.name)
        print("List of sub-processes: ", process_names)
        for process_name, process in self.processes.items():
            print("Current process: ", process_name)
            result = await process.ainvoke(*args, **updated_kwargs)
            output_kwargs = self._result_to_kwargs(result)
            if output_kwargs:
                output = ProcessOutput.model_validate(output_kwargs)
                updated_kwargs.update(output_kwargs)
            print("Process ", process_name, ": Completed")
        return output
