from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .sam_runner import validate_image_rgb
from .training import TinyUNet, keep_largest_connected_component


def load_trained_prep_model(model_path: str | Path, *, device: Optional[str] = None, base_channels: int = 16) -> TinyUNet:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyUNet(base_channels=base_channels)
    state_dict = torch.load(str(model_path), map_location=device_obj)
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    return model


def predict_prepared_tooth_mask(
    image_rgb,
    *,
    model,
    device: Optional[str] = None,
    image_size: int = 256,
    threshold: float = 0.5,
    largest_component_only: bool = True,
) -> np.ndarray:
    image_rgb = validate_image_rgb(image_rgb)
    orig_h, orig_w = image_rgb.shape[:2]

    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_tensor = image_tensor.to(device_obj)
    model = model.to(device_obj)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        pred = (torch.sigmoid(logits) >= threshold).float()
        if largest_component_only:
            pred = keep_largest_connected_component(pred).to(pred.device)
        mask_small = pred[0, 0].detach().cpu().numpy().astype(np.uint8)

    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask
