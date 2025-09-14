import os

from faceapp._base.fetcher import Fetcher


class LocalImageFetcher(Fetcher):

    async def fetch(self, path, *args, **kwargs):
        """
        Validate and return a local image path
        :param path:
        :return:
        """
        assert os.path.exists(path)
        return {"path": path, "meta": {}}
