import os
import mimetypes


# Allowed image MIME types
VALID_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff"
}


def is_valid_image(file_path):
    """Check if file has an image MIME type (based on extension)."""
    if not os.path.isfile(file_path):
        return False

    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in VALID_IMAGE_MIMETYPES
