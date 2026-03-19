import numpy as np

from tooth_service import taper
from tooth_service.taper import find_closest_angle, trapezoid_taper_no_rot


def test_find_closest_angle_clamps_to_supported_range():
    assert find_closest_angle(0.01) == 5
    assert find_closest_angle(0.90) == 28


def test_find_closest_angle_uses_actual_dictionary_bounds():
    original = taper.TAPER_DICT
    taper.TAPER_DICT = {3: 0.10, 9: 0.50}
    try:
        assert find_closest_angle(0.01) == 3
        assert find_closest_angle(0.90) == 9
    finally:
        taper.TAPER_DICT = original


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
    assert "lefts" not in result
    assert "rights" not in result
    assert "mask" not in result
    assert all(not isinstance(value, np.ndarray) for value in result.values())
