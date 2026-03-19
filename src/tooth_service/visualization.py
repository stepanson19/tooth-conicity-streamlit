from __future__ import annotations

import json
from typing import Iterable, Mapping

from .pipeline import serialize_pipeline_output


def results_to_rows(results: Iterable[Mapping[str, object]]):
    rows = []
    for item in results:
        rows.append(
            {
                "id": item.get("id"),
                "bbox_xyxy": item.get("bbox_xyxy"),
                "conicity_width_deg": item.get("conicity_width_deg"),
                "conicity_lr_deg": item.get("conicity_lr_deg"),
                "angle_from_dict": item.get("angle_from_dict"),
                "w_top": item.get("w_top"),
                "w_bot": item.get("w_bot"),
                "h_eff": item.get("h_eff"),
            }
        )
    return rows


def warning_lines(warnings: Iterable[str]):
    return [str(warning) for warning in warnings or []]


def status_message(output):
    status = output.get("status", "ok")
    error_stage = output.get("error_stage")
    if status == "error":
        return f"Status: error at {error_stage or 'analysis'}"
    if status == "empty":
        return "Status: no candidates after filtering"
    return "Status: analysis complete"


def download_filename(output):
    status = output.get("status", "ok")
    return f"tooth_results_{status}.json"


def download_payload(output):
    payload = serialize_pipeline_output(output)
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
