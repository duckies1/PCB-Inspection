"""Input image validation and preprocessing."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as a supported image."""


def decode_image(contents: bytes, content_type: str | None, allowed_types: set[str]) -> Image.Image:
    if content_type not in allowed_types:
        raise InvalidImageError("The uploaded file is not a supported image.")

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error

    return image.convert("RGB")
