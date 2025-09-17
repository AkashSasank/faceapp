import os

from faceapp._base.fetcher import Fetcher
from faceapp.utils.storage import s3


class LocalImageFetcher(Fetcher):

    async def fetch(self, path, *args, **kwargs):
        """
        Validate and return a local image path
        :param path:
        :return:
        """
        assert os.path.isfile(path)
        return {"path": path, "meta": {}}


class S3ImageFetcher(Fetcher):

    async def fetch(
        self, bucket_name: str, blob_name: str, download_dir: str, *args, **kwargs
    ):
        file_path = s3.download_file(bucket_name, blob_name, download_dir)
        assert os.path.exists(file_path)
        return {"path": file_path, "meta": {}}
