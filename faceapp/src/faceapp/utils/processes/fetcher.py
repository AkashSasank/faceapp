import os

from faceapp._base.fetcher import Fetcher
from faceapp.utils.misc import is_valid_image
from faceapp.utils.processes.process_outputs import FetchOutput
from faceapp.utils.storage import s3


class LocalImageFetcher(Fetcher):

    async def fetch(self, path, *args, **kwargs):
        """
        Validate and return a local image path
        :param path:
        :return:
        """
        assert os.path.isfile(path)
        provided_meta = kwargs.get("meta", {})
        if not isinstance(provided_meta, dict):
            provided_meta = {}
        if is_valid_image(path):
            return FetchOutput(path=path, meta=dict(provided_meta))
        return FetchOutput(path="", meta=dict(provided_meta))


class S3ImageFetcher(Fetcher):

    async def fetch(
        self,
        input_bucket: str,
        blob_name: str,
        download_dir: str,
        *args,
        **kwargs,
    ):
        file_path = s3.download_file(input_bucket, blob_name, download_dir)
        assert os.path.exists(file_path)
        provided_meta = kwargs.get("meta", {})
        if not isinstance(provided_meta, dict):
            provided_meta = {}
        storage_meta = {
            "storage_provider": "s3",
            "storage_bucket": input_bucket,
            "storage_object_key": blob_name,
        }
        return FetchOutput(
            path=file_path,
            meta={**provided_meta, **storage_meta},
        )
