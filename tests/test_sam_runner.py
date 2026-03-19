from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import tooth_service.sam_runner as sam_runner
from tooth_service.config import ensure_checkpoint_exists
from tooth_service.constants import DEFAULT_SAM_MODEL_TYPE


def test_resize_longest_leaves_small_image_unchanged():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    resized = sam_runner.resize_longest(image, max_side=200)

    assert resized.shape == image.shape


def test_resize_longest_scales_long_side_preserving_aspect():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = sam_runner.resize_longest(image, max_side=100)

    assert resized.shape == (50, 100, 3)


def test_validate_model_type_rejects_unknown_values():
    assert sam_runner.validate_model_type(DEFAULT_SAM_MODEL_TYPE) == DEFAULT_SAM_MODEL_TYPE

    with pytest.raises(ValueError, match="Unsupported SAM model type"):
        sam_runner.validate_model_type("bad-model")


def test_load_sam_model_uses_cpu_when_cuda_is_unavailable(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")

    model = FakeModel()
    seen = {}

    def factory(checkpoint):
        seen["checkpoint"] = checkpoint
        return model

    api = SimpleNamespace(sam_model_registry={"vit_h": factory})

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), raising=False)

    loaded = sam_runner.load_sam_model("vit_h", checkpoint)

    assert loaded is model
    assert seen["checkpoint"] == str(ensure_checkpoint_exists(checkpoint))
    assert model.device == "cpu"


def test_load_sam_model_falls_back_to_positional_device_for_keyword_mismatch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")

    model = KeywordOnlyDeviceModel()
    api = SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint: model})

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), raising=False)

    loaded = sam_runner.load_sam_model("vit_h", checkpoint)

    assert loaded is model
    assert model.calls == [("keyword", "cpu"), ("positional", "cpu")]


def test_load_sam_model_propagates_internal_type_error(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")

    model = InternalTypeErrorModel()
    api = SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint: model})

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), raising=False)

    with pytest.raises(TypeError, match="internal failure"):
        sam_runner.load_sam_model("vit_h", checkpoint)


def test_generate_masks_uses_autocast_on_cuda(monkeypatch):
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    model = FakeModel()
    mask_generator = FakeMaskGenerator()
    state = {"autocast_used": False}

    @contextmanager
    def fake_inference_mode():
        yield

    @contextmanager
    def fake_autocast(device_type, dtype=None):
        state["autocast_used"] = True
        assert device_type == "cuda"
        assert dtype == "float16"
        yield

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        inference_mode=fake_inference_mode,
        autocast=fake_autocast,
        float16="float16",
    )
    api = SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint: model}, SamAutomaticMaskGenerator=lambda model, **kwargs: mask_generator)

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", fake_torch, raising=False)

    masks = sam_runner.generate_masks(image, sam_model=model, device="cuda:0", max_side=100)

    assert state["autocast_used"] is True
    assert mask_generator.received_image.shape == image.shape
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0]["segmentation"], np.ones((2, 2), dtype=np.uint8))


def test_generate_masks_does_not_use_autocast_on_cpu(monkeypatch):
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    model = FakeModel()
    mask_generator = FakeMaskGenerator()

    @contextmanager
    def fake_inference_mode():
        yield

    def fake_autocast(*args, **kwargs):
        raise AssertionError("autocast should not be used on CPU")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        inference_mode=fake_inference_mode,
        autocast=fake_autocast,
        float16="float16",
    )
    api = SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint: model}, SamAutomaticMaskGenerator=lambda model, **kwargs: mask_generator)

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", fake_torch, raising=False)

    masks = sam_runner.generate_masks(image, sam_model=model, device="cpu", max_side=100)

    assert mask_generator.received_image.shape == image.shape
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0]["segmentation"], np.ones((2, 2), dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.zeros((24, 40), dtype=np.uint8),
        np.zeros((24, 40, 4), dtype=np.uint8),
        np.zeros((24, 40, 3), dtype=np.float32),
    ],
)
def test_generate_masks_rejects_malformed_image_inputs(monkeypatch, image):
    model = FakeModel()
    api = SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint: model}, SamAutomaticMaskGenerator=lambda model, **kwargs: FakeMaskGenerator())

    monkeypatch.setattr(sam_runner, "_get_segment_anything_api", lambda: api)
    monkeypatch.setattr(sam_runner, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), raising=False)

    with pytest.raises(ValueError, match="RGB uint8"):
        sam_runner.generate_masks(image, sam_model=model, device="cpu", max_side=100)


class FakeModel:
    def __init__(self):
        self.checkpoint = None
        self.device = None

    def to(self, device=None):
        self.device = device
        return self


class KeywordOnlyDeviceModel:
    def __init__(self):
        self.calls = []
        self.device = None

    def to(self, *args, **kwargs):
        if kwargs:
            self.calls.append(("keyword", kwargs["device"]))
            raise TypeError("to() got an unexpected keyword argument 'device'")
        self.calls.append(("positional", args[0]))
        self.device = args[0]
        return self


class InternalTypeErrorModel:
    def __init__(self):
        self.device = None

    def to(self, *args, **kwargs):
        raise TypeError("internal failure")


class FakeMaskGenerator:
    def __init__(self):
        self.received_image = None

    def generate(self, image):
        self.received_image = image
        return [{"segmentation": np.ones((2, 2), dtype=np.uint8)}]
