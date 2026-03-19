from __future__ import annotations

import importlib
from contextlib import contextmanager, nullcontext

import cv2
import numpy as np

from .config import ensure_checkpoint_exists
from .constants import DEFAULT_SAM_MODEL_TYPE, SAM_MODEL_TYPES

try:
    import torch as _torch
except Exception:  # pragma: no cover - optional dependency in local dev env
    _torch = None

torch = _torch


def _get_segment_anything_api():
    return importlib.import_module("segment_anything")


def validate_model_type(model_type: str) -> str:
    if model_type not in SAM_MODEL_TYPES:
        supported = ", ".join(SAM_MODEL_TYPES)
        raise ValueError(f"Unsupported SAM model type: {model_type!r}. Supported: {supported}")
    return model_type


def resize_longest(img, max_side=1536):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _torch_cuda_available() -> bool:
    if torch is None:
        return False
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _resolve_device(device=None) -> str:
    if device is None:
        return "cuda" if _torch_cuda_available() else "cpu"
    normalized = str(device)
    if normalized.startswith("cuda") and not _torch_cuda_available():
        return "cpu"
    return normalized


@contextmanager
def _inference_context(device: str):
    if torch is None:
        yield
        return

    inference_mode = getattr(torch, "inference_mode", None)
    base_context = inference_mode() if callable(inference_mode) else nullcontext()

    with base_context:
        if device.startswith("cuda"):
            autocast = getattr(torch, "autocast", None)
            float16 = getattr(torch, "float16", None)
            if callable(autocast):
                with autocast("cuda", dtype=float16):
                    yield
                return
        yield


def load_sam_model(model_type, checkpoint_path, device=None):
    model_type = validate_model_type(model_type)
    checkpoint = ensure_checkpoint_exists(checkpoint_path)
    api = _get_segment_anything_api()
    registry = getattr(api, "sam_model_registry", None)
    if registry is None or model_type not in registry:
        raise ValueError(f"SAM model registry does not contain {model_type!r}")

    resolved_device = _resolve_device(device)
    model = registry[model_type](checkpoint=str(checkpoint))
    if hasattr(model, "to"):
        try:
            model = model.to(device=resolved_device)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message or "device" not in message:
                raise
            model = model.to(resolved_device)
    return model


def validate_image_rgb(image_rgb):
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError("image_rgb must be an RGB uint8 ndarray with shape (H, W, 3)")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB uint8 ndarray with shape (H, W, 3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be an RGB uint8 ndarray with shape (H, W, 3)")
    return image_rgb


def generate_masks(
    image_rgb,
    sam_model=None,
    model_type=DEFAULT_SAM_MODEL_TYPE,
    checkpoint_path=None,
    device=None,
    max_side=1536,
    mask_generator_kwargs=None,
):
    validate_image_rgb(image_rgb)

    if sam_model is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when sam_model is not provided")
        sam_model = load_sam_model(model_type=model_type, checkpoint_path=checkpoint_path, device=device)

    api = _get_segment_anything_api()
    generator_cls = getattr(api, "SamAutomaticMaskGenerator", None)
    if generator_cls is None:
        raise AttributeError("segment_anything.SamAutomaticMaskGenerator is not available")

    generator_kwargs = dict(mask_generator_kwargs or {})
    mask_generator = generator_cls(model=sam_model, **generator_kwargs)
    image_small = resize_longest(image_rgb, max_side=max_side)
    runtime_device = _resolve_device(device if device is not None else getattr(sam_model, "device", None))

    with _inference_context(runtime_device):
        return mask_generator.generate(image_small)
