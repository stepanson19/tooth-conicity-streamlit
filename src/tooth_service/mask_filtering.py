from __future__ import annotations

import numpy as np

import cv2


def mask_props(seg: np.ndarray):
    seg_u8 = (seg.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = w * h
    if bbox_area == 0:
        return None

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = area / hull_area if hull_area > 0 else 0
    extent = area / bbox_area
    aspect = w / h if h > 0 else 0
    cy = y + h / 2.0
    return dict(
        area=area,
        x=x,
        y=y,
        w=w,
        h=h,
        cy=cy,
        bbox_area=bbox_area,
        solidity=solidity,
        extent=extent,
        aspect=aspect,
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def overlap_min(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    amin = min(a.sum(), b.sum())
    return float(inter) / float(amin + 1e-9)


def color_stats_hsv(hsv_img: np.ndarray, seg: np.ndarray):
    pix = hsv_img[seg.astype(bool)]
    if pix.size == 0:
        return dict(mean_h=0.0, mean_s=0.0, mean_v=0.0, orange_frac=0.0, dark_frac=1.0, white_frac=0.0)

    h = pix[:, 0].astype(np.float32)
    s = pix[:, 1].astype(np.float32)
    v = pix[:, 2].astype(np.float32)

    orange = (h >= 8) & (h <= 30) & (s >= 70) & (v >= 50)
    dark = v < 45
    tooth_white = (s < 85) & (v > 110)

    return dict(
        mean_h=float(h.mean()),
        mean_s=float(s.mean()),
        mean_v=float(v.mean()),
        orange_frac=float(orange.mean()),
        dark_frac=float(dark.mean()),
        white_frac=float(tooth_white.mean()),
    )


def select_instances(masks, iou_thresh=0.50, contain_thresh=0.85, max_instances=24):
    ms = sorted(masks, key=lambda m: (m.get("score", 0.0), m.get("props", {}).get("area", 0)), reverse=True)
    kept = []
    for m in ms:
        seg = m["segmentation"]
        ok = True
        for k in kept:
            if iou(seg, k["segmentation"]) > iou_thresh:
                ok = False
                break
            if overlap_min(seg, k["segmentation"]) > contain_thresh:
                ok = False
                break
        if ok:
            kept.append(m)
        if len(kept) >= max_instances:
            break
    return kept


def resize_mask(seg: np.ndarray, out_hw):
    out_h, out_w = out_hw
    if seg.shape == (out_h, out_w):
        return seg.astype(bool)
    seg_u8 = seg.astype(np.uint8)
    return cv2.resize(seg_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)


def crop_from_mask(img_rgb, seg, pad=12):
    ys, xs = np.where(seg > 0)
    if xs.size == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, int(x0 - pad))
    y0 = max(0, int(y0 - pad))
    x1 = min(img_rgb.shape[1] - 1, int(x1 + pad))
    y1 = min(img_rgb.shape[0] - 1, int(y1 + pad))
    crop = img_rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    mask_crop = seg[y0 : y1 + 1, x0 : x1 + 1].copy()
    return crop, mask_crop, (int(x0), int(y0), int(x1), int(y1))


def _raw_mask_score(mask, color):
    return (
        0.55 * float(mask.get("predicted_iou", 0.0))
        + 0.35 * float(mask.get("stability_score", 0.0))
        + 0.25 * color["white_frac"]
        + 0.15 * (1.0 - color["orange_frac"])
        + 0.10 * (1.0 - color["dark_frac"])
    )


def build_tooth_items(
    image_rgb,
    raw_masks,
    *,
    iou_thresh=0.50,
    contain_thresh=0.85,
    max_instances=24,
    pad=10,
):
    if not raw_masks:
        return [], []

    work_h, work_w = image_rgb.shape[:2]
    work_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    img_area = work_h * work_w

    filtered = []
    for m in raw_masks:
        seg = m["segmentation"].astype(bool)
        if seg.shape != (work_h, work_w):
            seg = resize_mask(seg, (work_h, work_w))

        props = mask_props(seg)
        if props is None:
            continue

        area = props["area"]
        if area < max(250, 0.0002 * img_area):
            continue
        if area > 0.08 * img_area:
            continue
        if props["w"] < 12 or props["h"] < 12:
            continue
        if props["h"] > 0.35 * work_h:
            continue
        if props["aspect"] < 0.25 or props["aspect"] > 4.5:
            continue
        if props["solidity"] < 0.50 or props["extent"] < 0.22:
            continue
        if props["cy"] < 0.20 * work_h or props["cy"] > 0.82 * work_h:
            continue

        color = color_stats_hsv(work_hsv, seg)
        if color["dark_frac"] > 0.25:
            continue
        if color["orange_frac"] > 0.32:
            continue
        if color["mean_v"] < 55:
            continue

        candidate = dict(m)
        candidate["segmentation"] = seg
        candidate["props"] = props
        candidate["color"] = color
        candidate["score"] = _raw_mask_score(candidate, color)
        filtered.append(candidate)

    instances_small = select_instances(filtered, iou_thresh=iou_thresh, contain_thresh=contain_thresh, max_instances=max_instances)

    instances = []
    for m in instances_small:
        m2 = dict(m)
        seg_orig = resize_mask(m["segmentation"], image_rgb.shape[:2])
        m2["segmentation"] = seg_orig
        p_orig = mask_props(seg_orig)
        if p_orig is not None:
            m2["props"] = p_orig
        instances.append(m2)

    tooth_items = []
    for idx, m in enumerate(instances):
        out = crop_from_mask(image_rgb, m["segmentation"].astype(np.uint8), pad=pad)
        if out is None:
            continue
        crop, mask_crop, bbox = out
        tooth_items.append(
            {
                "id": idx,
                "crop": crop,
                "mask_crop": mask_crop,
                "bbox": bbox,
                "segmentation": m["segmentation"],
            }
        )

    return tooth_items, instances
