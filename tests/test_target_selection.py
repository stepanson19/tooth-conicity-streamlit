import numpy as np

from tooth_service.target_selection import select_prepared_tooth


def test_select_prepared_tooth_prefers_centered_neutral_candidate():
    image = np.zeros((120, 220, 3), dtype=np.uint8)

    prepared_mask = make_mask((120, 220), (90, 30, 130, 95))
    natural_mask = make_mask((120, 220), (160, 22, 208, 98))

    image[prepared_mask.astype(bool)] = np.array([215, 210, 205], dtype=np.uint8)
    image[natural_mask.astype(bool)] = np.array([238, 222, 150], dtype=np.uint8)

    tooth_items = [
        {
            "id": 10,
            "bbox": (90, 30, 130, 95),
            "segmentation": prepared_mask,
            "mask_crop": prepared_mask[30:96, 90:131],
        },
        {
            "id": 11,
            "bbox": (160, 22, 208, 98),
            "segmentation": natural_mask,
            "mask_crop": natural_mask[22:99, 160:209],
        },
    ]
    results = [
        {"id": 10, "bbox_xyxy": [90, 30, 130, 95], "conicity_width_deg": 11.0},
        {"id": 11, "bbox_xyxy": [160, 22, 208, 98], "conicity_width_deg": 23.0},
    ]

    selected = select_prepared_tooth(image, tooth_items, results)

    assert selected is not None
    assert selected["candidate_count"] == 2
    assert selected["tooth_item"]["id"] == 10
    assert selected["result"]["id"] == 10
    assert selected["score"] > selected["candidates"][1]["score"]


def make_mask(shape, bbox):
    seg = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    seg[y0 : y1 + 1, x0 : x1 + 1] = 1
    return seg
