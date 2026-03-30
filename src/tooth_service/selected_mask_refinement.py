from __future__ import annotations

import cv2
import numpy as np

from .mask_filtering import crop_from_mask


def _connected_component_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8))
    if num_labels <= 1:
        return np.zeros_like(candidate, dtype=bool)

    keep = np.zeros_like(candidate, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & seed):
            keep |= component
    return keep


def refine_selected_tooth_item(image_rgb, tooth_item, *, pad=10):
    seg = np.asarray(tooth_item["segmentation"]).astype(bool)
    if not seg.any():
        return dict(tooth_item)

    x0, y0, x1, y1 = tooth_item["bbox"]
    height = max(1, y1 - y0 + 1)
    width = max(1, x1 - x0 + 1)
    image_h, image_w = image_rgb.shape[:2]

    pad_x = max(4, int(round(width * 0.18)))
    extend_y = max(8, int(round(height * 0.35)))
    roi_x0 = max(0, x0 - pad_x)
    roi_x1 = min(image_w - 1, x1 + pad_x)
    roi_y0 = max(0, y0 + int(round(height * 0.35)))
    roi_y1 = min(image_h - 1, y1 + extend_y)

    roi_seg = seg[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1]
    if not roi_seg.any():
        return dict(tooth_item)

    roi_img = image_rgb[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1]
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
    roi_lab = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    global_rows = np.arange(roi_y0, roi_y1 + 1)[:, None]
    bottom_band_start = y0 + int(round(height * 0.55))
    ref_mask = seg.copy()
    ref_mask[:bottom_band_start, :] = False
    if ref_mask.sum() < 16:
        ref_mask = seg

    ref_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[ref_mask]
    ref_mean = np.median(ref_lab, axis=0)
    delta = np.linalg.norm(roi_lab - ref_mean, axis=2)

    h = roi_hsv[:, :, 0].astype(np.float32)
    s = roi_hsv[:, :, 1].astype(np.float32)
    v = roi_hsv[:, :, 2].astype(np.float32)

    candidate = np.ones_like(roi_seg, dtype=bool)
    candidate &= ~roi_seg
    candidate &= global_rows >= y0 + int(round(height * 0.45))
    candidate &= delta <= 28.0
    candidate &= v >= max(90.0, float(np.percentile(v[roi_seg], 10)) - 25.0)
    candidate &= s <= 155.0
    candidate &= ~((h <= 12.0) & (s >= 150.0))

    seed = cv2.dilate(roi_seg.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    support = _connected_component_touching_seed(candidate, seed)

    if not support.any():
        return dict(tooth_item)

    merged = seg.copy()
    merged[roi_y0 : roi_y1 + 1, roi_x0 : roi_x1 + 1] |= support
    if merged.sum() <= seg.sum():
        return dict(tooth_item)

    out = crop_from_mask(image_rgb, merged.astype(np.uint8), pad=pad)
    if out is None:
        return dict(tooth_item)

    crop, mask_crop, bbox = out
    refined = dict(tooth_item)
    refined["segmentation"] = merged
    refined["crop"] = crop
    refined["mask_crop"] = mask_crop
    refined["bbox"] = bbox
    return refined
