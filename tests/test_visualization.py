import json

import numpy as np

from tooth_service import visualization
from tooth_service.visualization import download_payload, export_payload


def test_export_payload_strips_overlay_before_serialization(monkeypatch):
    captured = {}

    def fake_serialize_pipeline_output(output):
        captured["output"] = output
        return {"status": output.get("status", "ok")}

    monkeypatch.setattr(visualization, "serialize_pipeline_output", fake_serialize_pipeline_output)

    payload = {
        "status": "ok",
        "error": False,
        "error_stage": None,
        "instances_count": 1,
        "overlay_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "results": [{"id": 1, "bbox_xyxy": [1, 2, 3, 4]}],
        "warnings": [],
    }

    export_payload(payload)

    assert "overlay_image" not in captured["output"]


def test_download_payload_omits_overlay_image():
    payload = {
        "status": "ok",
        "error": False,
        "error_stage": None,
        "instances_count": 1,
        "overlay_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "results": [{"id": 1, "bbox_xyxy": [1, 2, 3, 4]}],
        "warnings": [],
    }

    exported = export_payload(payload)
    decoded = json.loads(download_payload(payload).decode("utf-8"))

    assert "overlay_image" not in exported
    assert "overlay_image" not in decoded
    assert decoded["results"][0]["id"] == 1
    assert decoded["status"] == "ok"
