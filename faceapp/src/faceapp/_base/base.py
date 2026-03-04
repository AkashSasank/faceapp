from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict


class ProcessOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


TProcessOutput = TypeVar("TProcessOutput", bound=ProcessOutput, covariant=True)


class Process(Protocol[TProcessOutput]):
    """
    Base protocall for all processes.
    All member classes in faceapp that execute a process
    is expected to follow this protocol.
    """

    async def ainvoke(self, *args, **kwargs) -> TProcessOutput: ...
