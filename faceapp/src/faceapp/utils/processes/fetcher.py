import os

from faceapp._base.fetcher import Fetcher
from faceapp.utils.storage import s3
from faceapp.utils.misc import is_valid_image


class LocalImageFetcher(Fetcher):

    async def fetch(self, path, *args, **kwargs):
        """
        Validate and return a local image path
        :param path:
        :return:
        """
        assert os.path.isfile(path)
        if is_valid_image(path):
            return {"path": path, "meta": {}}
        return {"path": "", "meta": {}}


class S3ImageFetcher(Fetcher):

    async def fetch(
        self, input_bucket: str, blob_name: str, download_dir: str, *args, **kwargs
    ):
        file_path = s3.download_file(input_bucket, blob_name, download_dir)
        assert os.path.exists(file_path)
        return {"path": file_path, "meta": {}}
