from __future__ import annotations

import numpy as np

from .constants import DEFAULT_BOT_Q, DEFAULT_SAM_MODEL_TYPE, DEFAULT_TOP_Q
from .mask_filtering import build_tooth_items
from .sam_runner import generate_masks, validate_image_rgb
from .taper import trapezoid_taper_no_rot
from .target_selection import select_prepared_tooth


def _overlay_segmentation_masks(image_rgb: np.ndarray, tooth_items, alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy()
    if not tooth_items:
        return overlay

    rng = np.random.default_rng(42)
    for tooth in tooth_items:
        seg = tooth.get("segmentation")
        if seg is None:
            continue
        seg_bool = np.asarray(seg).astype(bool)
        if not seg_bool.any():
            continue

        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay[seg_bool] = (overlay[seg_bool] * (1.0 - alpha) + color * alpha).astype(np.uint8)

    return overlay


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    return value


def serialize_results(results):
    return [_json_safe(result) for result in results]


def serialize_pipeline_output(output):
    return _json_safe(output)


def _pipeline_output(
    *,
    status,
    error_stage,
    warnings,
    results,
    overlay_image,
    instances_count,
    candidate_count=0,
    selected_tooth_id=None,
):
    return {
        "status": status,
        "error": status == "error",
        "error_stage": error_stage,
        "results": results,
        "overlay_image": overlay_image,
        "instances_count": instances_count,
        "candidate_count": candidate_count,
        "selected_tooth_id": selected_tooth_id,
        "warnings": warnings,
    }


def analyze_image(
    image_rgb,
    *,
    checkpoint_path=None,
    model_type=DEFAULT_SAM_MODEL_TYPE,
    sam_model=None,
    device=None,
    mask_generator_kwargs=None,
    top_q=DEFAULT_TOP_Q,
    bot_q=DEFAULT_BOT_Q,
    smooth=7,
    pad=10,
):
    warnings = []
    image_rgb = validate_image_rgb(image_rgb)

    try:
        raw_masks = generate_masks(
            image_rgb,
            sam_model=sam_model,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=device,
            mask_generator_kwargs=mask_generator_kwargs,
        )
    except Exception as exc:
        warnings.append(f"mask generation failed: {exc}")
        return _pipeline_output(
            status="error",
            error_stage="mask_generation",
            warnings=warnings,
            results=[],
            overlay_image=image_rgb.copy(),
            instances_count=0,
            candidate_count=0,
        )

    try:
        tooth_items, instances = build_tooth_items(image_rgb, raw_masks, pad=pad)
    except Exception as exc:
        warnings.append(f"tooth extraction failed: {exc}")
        return _pipeline_output(
            status="error",
            error_stage="tooth_extraction",
            warnings=warnings,
            results=[],
            overlay_image=image_rgb.copy(),
            instances_count=0,
            candidate_count=0,
        )

    if not tooth_items:
        warnings.append("No tooth candidates found after filtering")
        return _pipeline_output(
            status="empty",
            error_stage=None,
            warnings=warnings,
            results=[],
            overlay_image=image_rgb.copy(),
            instances_count=0,
            candidate_count=len(instances),
        )

    results = []
    for tooth in tooth_items:
        taper_result = None
        taper_exception = None
        try:
            taper_result = trapezoid_taper_no_rot(
                np.asarray(tooth["mask_crop"]).astype(np.uint8),
                top_q=top_q,
                bot_q=bot_q,
                smooth=smooth,
            )
        except Exception as exc:
            taper_exception = exc

        result = {
            "id": int(tooth["id"]),
            "bbox_xyxy": [int(coord) for coord in tooth["bbox"]],
            "angle_from_dict": None,
            "conicity_width_deg": None,
            "conicity_lr_deg": None,
            "w_top": None,
            "w_bot": None,
            "h_eff": None,
        }
        if taper_exception is not None:
            warnings.append(f"taper failed for tooth {tooth['id']}: {taper_exception}")
        elif taper_result is None:
            warnings.append(f"taper could not be computed for tooth {tooth['id']}")
        else:
            result.update(
                {
                    "angle_from_dict": taper_result.get("angle_from_dict"),
                    "conicity_width_deg": taper_result.get("conicity_width_deg"),
                    "conicity_lr_deg": taper_result.get("conicity_lr_deg"),
                    "w_top": taper_result.get("w_top"),
                    "w_bot": taper_result.get("w_bot"),
                    "h_eff": taper_result.get("h_eff"),
                }
            )
        results.append(result)

    selection = select_prepared_tooth(image_rgb, tooth_items, results)
    if selection is None:
        warnings.append("No prepared tooth candidate could be selected")
        return _pipeline_output(
            status="empty",
            error_stage=None,
            warnings=warnings,
            results=[],
            overlay_image=image_rgb.copy(),
            instances_count=0,
            candidate_count=len(instances),
        )

    selected_tooth = selection["tooth_item"]
    selected_result = selection["result"]
    candidate_count = int(selection["candidate_count"])

    return _pipeline_output(
        status="ok",
        error_stage=None,
        warnings=warnings,
        results=[selected_result],
        overlay_image=_overlay_segmentation_masks(image_rgb, [selected_tooth]),
        instances_count=1,
        candidate_count=candidate_count,
        selected_tooth_id=int(selected_result["id"]),
    )
