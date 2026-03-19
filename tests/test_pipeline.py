import numpy as np

from tooth_service import pipeline


def test_analyze_image_returns_structured_results(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    tooth_items = [
        {
            "id": 7,
            "crop": image.copy(),
            "mask_crop": mask.copy(),
            "bbox": (4, 4, 11, 11),
            "segmentation": mask.copy(),
        }
    ]
    instances = [{"segmentation": mask.copy(), "score": 0.99}]

    monkeypatch.setattr(
        pipeline,
        "generate_masks",
        lambda *args, **kwargs: [{"segmentation": mask.copy(), "score": 0.99}],
    )
    monkeypatch.setattr(pipeline, "build_tooth_items", lambda *args, **kwargs: (tooth_items, instances))
    monkeypatch.setattr(
        pipeline,
        "trapezoid_taper_no_rot",
        lambda *args, **kwargs: {
            "angle_from_dict": 10,
            "conicity_width_deg": 10.0,
            "conicity_lr_deg": 12.5,
            "w_top": 1.0,
            "w_bot": 2.0,
            "h_eff": 3.0,
        },
    )

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["instances_count"] == 1
    assert result["warnings"] == []
    assert result["overlay_image"].shape == image.shape
    assert result["results"][0]["id"] == 7
    assert result["results"][0]["bbox_xyxy"] == [4, 4, 11, 11]
    assert result["results"][0]["conicity_width_deg"] == 10.0


def test_analyze_image_records_warning_when_taper_fails(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    tooth_items = [
        {
            "id": 3,
            "crop": image.copy(),
            "mask_crop": mask.copy(),
            "bbox": (4, 4, 11, 11),
            "segmentation": mask.copy(),
        }
    ]

    monkeypatch.setattr(
        pipeline,
        "generate_masks",
        lambda *args, **kwargs: [{"segmentation": mask.copy(), "score": 0.99}],
    )
    monkeypatch.setattr(pipeline, "build_tooth_items", lambda *args, **kwargs: (tooth_items, [{"segmentation": mask.copy()}]))
    monkeypatch.setattr(pipeline, "trapezoid_taper_no_rot", lambda *args, **kwargs: None)

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["instances_count"] == 1
    assert len(result["warnings"]) == 1
    assert "taper" in result["warnings"][0].lower()
    assert result["results"][0]["conicity_width_deg"] is None
    assert result["results"][0]["conicity_lr_deg"] is None


def test_serialize_results_normalizes_numpy_and_tuples():
    results = [
        {
            "id": np.int64(7),
            "bbox_xyxy": (1, 2, 3, 4),
            "conicity_width_deg": np.float32(9.5),
        }
    ]

    serialized = pipeline.serialize_results(results)

    assert serialized == [
        {
            "id": 7,
            "bbox_xyxy": [1, 2, 3, 4],
            "conicity_width_deg": 9.5,
        }
    ]
