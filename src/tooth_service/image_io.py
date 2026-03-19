from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB uint8 array."""
    if not file_bytes:
        raise ValueError("Uploaded file is not a valid image")

    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    try:
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    if image_bgr is None:
        raise ValueError("Uploaded file is not a valid image")

    try:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as exc:
        raise ValueError("Uploaded file is not a valid image") from exc
