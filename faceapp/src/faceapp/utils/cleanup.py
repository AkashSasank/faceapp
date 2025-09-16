from faceapp._base.base import Process
import os
import shutil


class FileCleanup(Process):

    async def ainvoke(self, input_path: str, output_path: str, *args, **kwargs) -> dict:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if os.path.exists(input_path):
            shutil.move(input_path, output_path)
        return {"cleanup_status": True}
