import numpy as np

from tooth_service.mask_filtering import build_tooth_items, crop_from_mask, mask_props


def test_mask_props_returns_bbox_metrics():
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[4:15, 6:12] = 1

    props = mask_props(seg)

    assert props is not None
    assert props["area"] > 0
    assert props["w"] > 0
    assert props["h"] > 0
    assert props["bbox_area"] == props["w"] * props["h"]


def test_crop_from_mask_returns_crop_mask_and_bbox():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[5:10, 7:11] = 1

    crop, mask_crop, bbox = crop_from_mask(img, seg, pad=0)

    assert crop.shape[:2] == mask_crop.shape
    assert bbox == (7, 5, 10, 9)
    assert mask_crop.sum() == seg.sum()


def test_build_tooth_items_keeps_non_overlapping_instances():
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    raw_masks = [
        {"segmentation": make_mask((80, 80), (5, 18, 25, 40)), "score": 0.95, "predicted_iou": 0.92, "stability_score": 0.88},
        {"segmentation": make_mask((80, 80), (40, 10, 60, 32)), "score": 0.90, "predicted_iou": 0.85, "stability_score": 0.86},
    ]

    tooth_items, instances = build_tooth_items(image, raw_masks, pad=0, max_instances=10)

    assert len(instances) == 2
    assert len(tooth_items) == 2
    assert tooth_items[0]["id"] == 0
    assert tooth_items[0]["crop"].shape[:2] == tooth_items[0]["mask_crop"].shape
    assert tooth_items[0]["bbox"] == (5, 18, 25, 40)
    assert tooth_items[0]["segmentation"].shape == image.shape[:2]
    assert "segmentation" in instances[0]
    assert instances[0]["segmentation"].shape == image.shape[:2]


def make_mask(shape, bbox):
    seg = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    seg[y0 : y1 + 1, x0 : x1 + 1] = 1
    return seg
