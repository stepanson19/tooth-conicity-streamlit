from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _masked_hsv_stats(image_rgb: np.ndarray, seg: np.ndarray) -> dict[str, float]:
    seg_bool = np.asarray(seg).astype(bool)
    if not seg_bool.any():
        return {
            "mean_s": 255.0,
            "mean_v": 0.0,
            "orange_frac": 1.0,
            "white_frac": 0.0,
        }

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    pix = hsv[seg_bool]
    h = pix[:, 0].astype(np.float32)
    s = pix[:, 1].astype(np.float32)
    v = pix[:, 2].astype(np.float32)
    orange = (h >= 8) & (h <= 30) & (s >= 55) & (v >= 70)
    white = (s < 70) & (v > 120)
    return {
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "orange_frac": float(orange.mean()),
        "white_frac": float(white.mean()),
    }


def _bbox_center_score(bbox, image_width: int) -> float:
    x0, _, x1, _ = bbox
    bbox_cx = (float(x0) + float(x1)) / 2.0
    image_cx = (float(image_width) - 1.0) / 2.0
    dist = abs(bbox_cx - image_cx) / max(1.0, image_width / 2.0)
    return _clip01(1.0 - dist)


def _edge_clearance_score(bbox, image_shape) -> float:
    image_h, image_w = image_shape[:2]
    x0, y0, x1, y1 = bbox
    clear_left = max(0.0, float(x0))
    clear_top = max(0.0, float(y0))
    clear_right = max(0.0, float(image_w - 1 - x1))
    clear_bottom = max(0.0, float(image_h - 1 - y1))
    shortest = min(clear_left, clear_top, clear_right, clear_bottom)
    target_margin = min(image_h, image_w) * 0.08
    return _clip01(shortest / max(1.0, target_margin))


def _size_score(seg: np.ndarray, image_shape) -> float:
    image_area = float(image_shape[0] * image_shape[1])
    area_ratio = float(np.asarray(seg).astype(bool).sum()) / max(1.0, image_area)
    if area_ratio <= 0.0:
        return 0.0
    if area_ratio < 0.01:
        return _clip01(area_ratio / 0.01)
    if area_ratio > 0.15:
        return _clip01(1.0 - ((area_ratio - 0.15) / 0.10))
    return 1.0


def _conicity_score(result: Mapping[str, object]) -> float:
    conicity = result.get("conicity_width_deg")
    if conicity is None:
        return 0.35
    value = float(conicity)
    return _clip01(1.0 - abs(value - 14.0) / 18.0)


def _prepared_color_score(stats: Mapping[str, float]) -> float:
    low_orange = 1.0 - float(stats["orange_frac"])
    low_saturation = 1.0 - min(1.0, float(stats["mean_s"]) / 140.0)
    bright_enough = _clip01((float(stats["mean_v"]) - 70.0) / 120.0)
    return _clip01(0.50 * low_orange + 0.30 * low_saturation + 0.20 * bright_enough)


def _score_candidate(image_rgb: np.ndarray, tooth_item: Mapping[str, object], result: Mapping[str, object]) -> dict[str, object]:
    bbox = tooth_item["bbox"]
    seg = np.asarray(tooth_item["segmentation"]).astype(bool)
    color_stats = _masked_hsv_stats(image_rgb, seg)
    center_score = _bbox_center_score(bbox, image_rgb.shape[1])
    edge_score = _edge_clearance_score(bbox, image_rgb.shape)
    color_score = _prepared_color_score(color_stats)
    size_score = _size_score(seg, image_rgb.shape)
    conicity_score = _conicity_score(result)

    score = (
        0.35 * center_score
        + 0.20 * edge_score
        + 0.20 * color_score
        + 0.15 * size_score
        + 0.10 * conicity_score
    )

    return {
        "tooth_id": int(tooth_item["id"]),
        "score": float(score),
        "center_score": float(center_score),
        "edge_score": float(edge_score),
        "color_score": float(color_score),
        "size_score": float(size_score),
        "conicity_score": float(conicity_score),
    }


def select_prepared_tooth(
    image_rgb: np.ndarray,
    tooth_items: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
):
    if not tooth_items or not results:
        return None

    results_by_id = {int(item["id"]): item for item in results}
    candidates = []
    for tooth_item in tooth_items:
        tooth_id = int(tooth_item["id"])
        result = results_by_id.get(tooth_id)
        if result is None:
            continue
        candidate = _score_candidate(image_rgb, tooth_item, result)
        candidate["tooth_item"] = tooth_item
        candidate["result"] = result
        candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    return {
        "candidate_count": len(candidates),
        "tooth_item": best["tooth_item"],
        "result": best["result"],
        "score": best["score"],
        "candidates": [
            {
                "tooth_id": candidate["tooth_id"],
                "score": candidate["score"],
                "center_score": candidate["center_score"],
                "edge_score": candidate["edge_score"],
                "color_score": candidate["color_score"],
                "size_score": candidate["size_score"],
                "conicity_score": candidate["conicity_score"],
            }
            for candidate in candidates
        ],
    }
