import os
import shutil

from faceapp._base.base import Process
from faceapp.utils.storage import s3


class LocalFileCleanup(Process):

    async def ainvoke(self, input_path: str, output_path: str, *args, **kwargs) -> dict:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if os.path.exists(input_path):
            shutil.move(input_path, output_path)
        return {"cleanup_status": True}


class S3Cleanup(Process):
    async def ainvoke(
        self, input_bucket: str, output_bucket: str, blob_name: str, *args, **kwargs
    ) -> dict:
        status = s3.move_file(
            source_bucket=input_bucket, source_key=blob_name, dest_bucket=output_bucket
        )
        return {"cleanup_status": status}
