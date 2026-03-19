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
    assert result["status"] == "ok"
    assert result["error"] is False
    assert result["error_stage"] is None
    assert result["warnings"] == []
    assert result["overlay_image"].shape == image.shape
    assert result["results"][0]["id"] == 7
    assert result["results"][0]["bbox_xyxy"] == [4, 4, 11, 11]
    assert result["results"][0]["conicity_width_deg"] == 10.0


def test_analyze_image_records_single_warning_when_taper_fails(monkeypatch):
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
    monkeypatch.setattr(
        pipeline,
        "trapezoid_taper_no_rot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["instances_count"] == 1
    assert result["status"] == "ok"
    assert result["error"] is False
    assert result["error_stage"] is None
    assert result["warnings"] == ["taper failed for tooth 3: boom"]
    assert result["results"][0]["conicity_width_deg"] is None
    assert result["results"][0]["conicity_lr_deg"] is None


def test_analyze_image_marks_mask_generation_failure(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(
        pipeline,
        "generate_masks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sam failed")),
    )

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["status"] == "error"
    assert result["error"] is True
    assert result["error_stage"] == "mask_generation"
    assert result["results"] == []
    assert result["instances_count"] == 0
    assert len(result["warnings"]) == 1
    assert "sam failed" in result["warnings"][0]


def test_analyze_image_marks_tooth_extraction_failure(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    monkeypatch.setattr(
        pipeline,
        "generate_masks",
        lambda *args, **kwargs: [{"segmentation": mask.copy(), "score": 0.99}],
    )
    monkeypatch.setattr(
        pipeline,
        "build_tooth_items",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("filter failed")),
    )

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["status"] == "error"
    assert result["error"] is True
    assert result["error_stage"] == "tooth_extraction"
    assert result["results"] == []
    assert result["instances_count"] == 0
    assert len(result["warnings"]) == 1
    assert "filter failed" in result["warnings"][0]


def test_analyze_image_marks_empty_outcome_without_error(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline, "generate_masks", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "build_tooth_items", lambda *args, **kwargs: ([], []))

    result = pipeline.analyze_image(image, checkpoint_path="unused.pth", sam_model=object(), model_type="vit_h")

    assert result["status"] == "empty"
    assert result["error"] is False
    assert result["error_stage"] is None
    assert result["results"] == []
    assert result["instances_count"] == 0
    assert result["warnings"] == ["No tooth candidates found after filtering"]


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


def test_serialize_pipeline_output_normalizes_numpy_payload():
    payload = {
        "status": "ok",
        "error": False,
        "error_stage": None,
        "instances_count": np.int64(1),
        "overlay_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "results": [
            {
                "id": np.int64(3),
                "bbox_xyxy": (1, 2, 3, 4),
                "conicity_width_deg": np.float32(9.5),
            }
        ],
        "warnings": ["x"],
    }

    serialized = pipeline.serialize_pipeline_output(payload)

    assert serialized["instances_count"] == 1
    assert serialized["overlay_image"] == [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    assert serialized["results"][0]["id"] == 3
    assert serialized["results"][0]["bbox_xyxy"] == [1, 2, 3, 4]
