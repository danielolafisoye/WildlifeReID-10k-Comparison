"""Image validation and model-specific preprocessing (steps 1-2 of the inference pipeline)."""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

# ImageNet statistics, used by both MiewID and ConvNeXt pretrained weights.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InvalidImageError(ValueError):
    """Raised when the upload is not a readable image."""


def load_image(data: bytes) -> Image.Image:
    """Validate the bytes as an image and return it as RGB.

    The WildlifeReID-10k images are already cropped around the animal, so no detector is run here.
    """
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()  # catches truncated / corrupt files
        image = Image.open(io.BytesIO(data))  # verify() invalidates the handle, so reopen
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc


def to_chw_array(image: Image.Image, size: int) -> np.ndarray:
    """Resize to size x size, scale to [0, 1], apply ImageNet normalisation, return CHW float32."""
    resized = image.resize((size, size), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))
