from abc import ABC
from typing import Protocol, Union

from faceapp._base.indexer import Indexer


class Search(ABC):

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def search(self, *args, **kwargs) -> Union[list, dict]:
        return self.indexer.search(*args, **kwargs)
