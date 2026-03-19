import numpy as np

from tooth_service.taper import find_closest_angle, trapezoid_taper_no_rot


def test_find_closest_angle_clamps_to_supported_range():
    assert find_closest_angle(0.01) == 5
    assert find_closest_angle(0.90) == 28


def test_trapezoid_taper_returns_normalized_result_schema():
    mask = np.zeros((40, 20), dtype=np.uint8)
    for y in range(5, 35):
        left = 6 - min((y - 5) // 10, 2)
        right = 13 + min((y - 5) // 10, 2)
        mask[y, left : right + 1] = 1

    result = trapezoid_taper_no_rot(mask, top_q=0.15, bot_q=0.65, smooth=1)

    assert result is not None
    assert "angle_from_dict" in result
    assert "conicity_width_deg" in result
    assert result["conicity_width_deg"] == result["angle_from_dict"]
    assert "conicity_lr_deg" in result
