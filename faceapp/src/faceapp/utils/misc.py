import mimetypes
import os

try:
    import magic  # type: ignore[import-not-found]
except (ImportError, OSError):
    magic = None


def is_valid_image(file_path):
    """Check if file is an image using MIME type detection."""
    if not os.path.isfile(file_path):
        return False

    mime_type = None

    if magic is not None:
        try:
            mime_type = magic.from_file(file_path, mime=True)
        except OSError:
            mime_type = None

    if not mime_type:
        guessed_type, _ = mimetypes.guess_type(file_path)
        mime_type = guessed_type

    return isinstance(mime_type, str) and mime_type.startswith("image/")
