import numpy as np

from tooth_service.selected_mask_refinement import refine_selected_tooth_item


def test_refine_selected_tooth_item_expands_into_connected_cervical_region():
    image = np.zeros((120, 140, 3), dtype=np.uint8)
    image[:, :] = np.array([210, 32, 18], dtype=np.uint8)

    seg = np.zeros((120, 140), dtype=np.uint8)
    seg[20:68, 50:90] = 1

    image[seg.astype(bool)] = np.array([228, 206, 176], dtype=np.uint8)
    image[66:82, 44:96] = np.array([239, 220, 198], dtype=np.uint8)

    tooth_item = {
        "id": 0,
        "bbox": (50, 20, 89, 67),
        "segmentation": seg.astype(bool),
        "crop": image[20:68, 50:90].copy(),
        "mask_crop": seg[20:68, 50:90].copy(),
    }

    refined = refine_selected_tooth_item(image, tooth_item, pad=0)

    assert refined["bbox"] == (44, 20, 95, 81)
    assert refined["segmentation"].sum() > tooth_item["segmentation"].sum()


def test_refine_selected_tooth_item_does_not_flood_plain_gum():
    image = np.zeros((120, 140, 3), dtype=np.uint8)
    image[:, :] = np.array([210, 32, 18], dtype=np.uint8)

    seg = np.zeros((120, 140), dtype=np.uint8)
    seg[20:68, 50:90] = 1
    image[seg.astype(bool)] = np.array([228, 206, 176], dtype=np.uint8)

    tooth_item = {
        "id": 0,
        "bbox": (50, 20, 89, 67),
        "segmentation": seg.astype(bool),
        "crop": image[20:68, 50:90].copy(),
        "mask_crop": seg[20:68, 50:90].copy(),
    }

    refined = refine_selected_tooth_item(image, tooth_item, pad=0)

    assert refined["bbox"] == tooth_item["bbox"]
    assert refined["segmentation"].sum() == tooth_item["segmentation"].sum()
