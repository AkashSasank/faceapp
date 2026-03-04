import os

import magic


def is_valid_image(file_path):
    """Check if file is an image using MIME type detection."""
    if not os.path.isfile(file_path):
        return False

    mime_type = magic.from_file(file_path, mime=True)
    return isinstance(mime_type, str) and mime_type.startswith("image/")
