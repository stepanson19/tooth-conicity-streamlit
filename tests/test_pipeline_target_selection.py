import numpy as np

from tooth_service import pipeline


def test_analyze_image_returns_only_selected_prepared_tooth(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    left_mask = np.zeros((20, 20), dtype=np.uint8)
    left_mask[4:12, 2:8] = 1
    center_mask = np.zeros((20, 20), dtype=np.uint8)
    center_mask[4:12, 9:15] = 1

    tooth_items = [
        {
            "id": 1,
            "crop": image.copy(),
            "mask_crop": left_mask[4:12, 2:8].copy(),
            "bbox": (2, 4, 7, 11),
            "segmentation": left_mask.copy(),
        },
        {
            "id": 2,
            "crop": image.copy(),
            "mask_crop": center_mask[4:12, 9:15].copy(),
            "bbox": (9, 4, 14, 11),
            "segmentation": center_mask.copy(),
        },
    ]
    instances = [{"segmentation": left_mask.copy()}, {"segmentation": center_mask.copy()}]

    monkeypatch.setattr(
        pipeline,
        "generate_masks",
        lambda *args, **kwargs: [{"segmentation": left_mask.copy()}, {"segmentation": center_mask.copy()}],
    )
    monkeypatch.setattr(pipeline, "build_tooth_items", lambda *args, **kwargs: (tooth_items, instances))

    taper_values = {
        1: {"angle_from_dict": 25, "conicity_width_deg": 25.0, "conicity_lr_deg": 30.0, "w_top": 1.0, "w_bot": 2.0, "h_eff": 3.0},
        2: {"angle_from_dict": 12, "conicity_width_deg": 12.0, "conicity_lr_deg": 20.0, "w_top": 1.5, "w_bot": 2.5, "h_eff": 4.0},
    }

    def _fake_taper(mask_crop, **kwargs):
        width = mask_crop.shape[1]
        return taper_values[1 if width == 6 else 2]

    monkeypatch.setattr(pipeline, "trapezoid_taper_no_rot", _fake_taper)
    monkeypatch.setattr(
        pipeline,
        "select_prepared_tooth",
        lambda image_rgb, tooth_items, results: {
            "candidate_count": 2,
            "tooth_item": tooth_items[1],
            "result": results[1],
            "score": 0.88,
            "candidates": [
                {"tooth_id": 2, "score": 0.88},
                {"tooth_id": 1, "score": 0.41},
            ],
        },
    )

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["status"] == "ok"
    assert result["error"] is False
    assert result["candidate_count"] == 2
    assert result["instances_count"] == 1
    assert result["selected_tooth_id"] == 2
    assert [item["id"] for item in result["results"]] == [2]
