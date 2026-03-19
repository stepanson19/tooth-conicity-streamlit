import cv2
import numpy as np
import pytest

from tooth_service.image_io import decode_uploaded_image


def test_decode_uploaded_image_returns_rgb_uint8_array():
    image_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    image_bgr[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".png", image_bgr)
    assert ok

    decoded = decode_uploaded_image(encoded.tobytes())

    assert decoded.shape == (4, 5, 3)
    assert decoded.dtype == np.uint8
    assert tuple(decoded[0, 0]) == (30, 20, 10)


def test_decode_uploaded_image_rejects_empty_bytes():
    with pytest.raises(ValueError, match="valid image"):
        decode_uploaded_image(b"")


def test_decode_uploaded_image_rejects_invalid_bytes():
    with pytest.raises(ValueError, match="valid image"):
        decode_uploaded_image(b"not an image")
