from typing import Protocol

from faceapp._base.indexer import Indexer


class Search(Protocol):

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def search(self, *args, **kwargs) -> [list, dict]:
        return self.indexer.search(*args, **kwargs)
