from pathlib import Path
from typing import cast
import cv2 as cv
import numpy as np
import numpy.typing as npt

GrayImage = npt.NDArray[np.uint8]

def upscaledOCR(image: GrayImage) -> GrayImage:
    """Upscale one grayscale image for OCR and apply CLAHE."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a NumPy array with dtype uint8")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty grayscale image")
    height, width = image.shape
    if height >= 4840 or width >= 3428:
        # OCR preprocessing only upscales; leave images outside the existing
        # upscaling bounds unchanged.
        return image
    data_dir = Path(__file__).parent
    model_path = data_dir / "ESPCN_x2.pb"
    if not model_path.is_file():
        raise FileNotFoundError(f"Super-resolution model not found: {model_path}")
    sr = cv.dnn_superres.DnnSuperResImpl.create()
    sr.readModel(str(model_path))
    sr.setModel("espcn", 2)
    upscaled = sr.upsample(image)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return cast(GrayImage, clahe.apply(upscaled))

def downscaledSegmentation(image: GrayImage) -> GrayImage:
    """Downscale one grayscale image so neither dimension exceeds 1024."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a NumPy array with dtype uint8")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty grayscale image")
    height, width = image.shape
    max_dimension = 1024
    if height <= max_dimension and width <= max_dimension:
        return image
    scale = max_dimension / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cast(
        GrayImage,
        cv.resize (
            image,
            (resized_width, resized_height),
            interpolation=cv.INTER_AREA,
        ),
    )
