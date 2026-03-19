import importlib

from tooth_service.constants import DEFAULT_TOP_Q, DEFAULT_BOT_Q


def test_default_thresholds_are_exposed():
    assert DEFAULT_TOP_Q == 0.15
    assert DEFAULT_BOT_Q == 0.65


def test_segment_anything_imports():
    mod = importlib.import_module("segment_anything")
    assert mod is not None
