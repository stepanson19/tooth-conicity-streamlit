from __future__ import annotations

import math

import numpy as np

from .constants import TAPER_DICT


def find_closest_angle(ratio):
    """Return the closest supported taper angle for a width ratio."""
    if not TAPER_DICT:
        raise ValueError("TAPER_DICT must not be empty")

    angles = sorted(TAPER_DICT)
    min_angle = angles[0]
    max_angle = angles[-1]
    if ratio <= TAPER_DICT[min_angle]:
        return min_angle
    if ratio >= TAPER_DICT[max_angle]:
        return max_angle
    return min(angles, key=lambda angle: abs(TAPER_DICT[angle] - ratio))


def widths_by_row(mask01: np.ndarray):
    """For each row, compute left/right boundaries and width of the mask."""
    h, _ = mask01.shape
    widths = np.zeros(h, dtype=np.int32)
    lefts = np.full(h, -1, dtype=np.int32)
    rights = np.full(h, -1, dtype=np.int32)
    for y in range(h):
        xs = np.where(mask01[y, :] > 0)[0]
        if xs.size:
            lefts[y] = int(xs.min())
            rights[y] = int(xs.max())
            widths[y] = rights[y] - lefts[y] + 1
    return lefts, rights, widths


def trapezoid_taper_no_rot(mask01: np.ndarray, top_q=0.25, bot_q=0.80, smooth=7):
    """
    Estimate taper on a vertically oriented tooth mask without rotation.
    Returns a normalized schema with both `angle_from_dict` and
    `conicity_width_deg` for downstream consumers.
    """
    lefts, rights, widths = widths_by_row(mask01)
    ys = np.where(widths > 0)[0]
    if ys.size < 10:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    height = y1 - y0 + 1

    w_s = widths.astype(np.float32)
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        w_s_valid = np.convolve(w_s[ys], k, mode="same")
        w_s[ys] = w_s_valid

    y_top = int(y0 + top_q * height)
    y_bot = int(y0 + bot_q * height)
    y_top = int(np.clip(y_top, y0, y1))
    y_bot = int(np.clip(y_bot, y0, y1))
    if y_bot <= y_top:
        return None

    w_top = float(w_s[y_top])
    w_bot = float(w_s[y_bot])
    h_eff = float(y_bot - y_top)

    delta = abs(w_bot - w_top)
    ratio = delta / (2.0 * h_eff + 1e-9)
    angle_from_dict = find_closest_angle(ratio)

    ys_fit = np.arange(y_top, y_bot + 1)
    xL = lefts[ys_fit]
    xR = rights[ys_fit]
    valid = (xL >= 0) & (xR >= 0)
    ys_fit = ys_fit[valid]
    xL = xL[valid].astype(np.float32)
    xR = xR[valid].astype(np.float32)

    left_wall_deg = None
    right_wall_deg = None
    conicity_lr_deg = None
    if ys_fit.size >= 8:
        aL, _ = np.polyfit(ys_fit, xL, 1)
        aR, _ = np.polyfit(ys_fit, xR, 1)
        left_wall_deg = math.degrees(math.atan(abs(aL)))
        right_wall_deg = math.degrees(math.atan(abs(aR)))
        conicity_lr_deg = left_wall_deg + right_wall_deg

    return {
        "y_top": y_top,
        "y_bot": y_bot,
        "w_top": w_top,
        "w_bot": w_bot,
        "h_eff": h_eff,
        "ratio": ratio,
        "angle_from_dict": angle_from_dict,
        "conicity_width_deg": float(angle_from_dict),
        "left_wall_deg": left_wall_deg,
        "right_wall_deg": right_wall_deg,
        "conicity_lr_deg": conicity_lr_deg,
    }
