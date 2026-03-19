from pathlib import Path

import pytest

from tooth_service.config import ensure_checkpoint_exists, resolve_checkpoint_path


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
