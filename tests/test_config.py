from pathlib import Path

import pytest

from tooth_service import config
from tooth_service.config import default_checkpoint_path, ensure_checkpoint_exists, resolve_checkpoint_path


def test_resolve_checkpoint_path_prefers_existing_file(tmp_path):
    ckpt = tmp_path / "sam_vit_h_4b8939.pth"
    ckpt.write_bytes(b"x")

    resolved = resolve_checkpoint_path(ckpt)

    assert resolved == ckpt


def test_ensure_checkpoint_exists_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        ensure_checkpoint_exists(missing)


def test_ensure_checkpoint_exists_rejects_directory(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        ensure_checkpoint_exists(directory)


def test_ensure_checkpoint_exists_rejects_incomplete_known_checkpoint(tmp_path):
    partial = tmp_path / "sam_vit_h_4b8939.pth"
    partial.write_bytes(b"x" * 1024)

    with pytest.raises(ValueError, match="incomplete"):
        ensure_checkpoint_exists(partial)


def test_default_checkpoint_path_uses_model_filename(tmp_path):
    resolved = default_checkpoint_path(tmp_path, model_type="vit_b")

    assert resolved == tmp_path / "checkpoints" / "sam_vit_b_01ec64.pth"


def test_ensure_checkpoint_exists_downloads_missing_known_checkpoint_when_enabled(monkeypatch, tmp_path):
    target = tmp_path / "sam_vit_b_01ec64.pth"
    monkeypatch.setenv(config.AUTO_DOWNLOAD_CHECKPOINT_ENV, "1")
    monkeypatch.setitem(config.KNOWN_CHECKPOINT_SIZES, target.name, 4)
    monkeypatch.setitem(config.KNOWN_CHECKPOINT_URLS, target.name, "https://example.invalid/sam_vit_b_01ec64.pth")

    calls = {}

    def fake_download(url, destination):
        calls["url"] = url
        destination.write_bytes(b"test")

    monkeypatch.setattr(config, "_download_file", fake_download)

    resolved = ensure_checkpoint_exists(target)

    assert resolved == target.resolve()
    assert calls["url"] == "https://example.invalid/sam_vit_b_01ec64.pth"
