from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

KNOWN_CHECKPOINT_SIZES = {
    "sam_vit_h_4b8939.pth": 2564550879,
    "sam_vit_l_0b3195.pth": 1249524607,
    "sam_vit_b_01ec64.pth": 375042383,
}


def resolve_checkpoint_path(path_like: PathLike) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve()


def ensure_checkpoint_exists(path_like: PathLike) -> Path:
    path = resolve_checkpoint_path(path_like)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    expected_size = KNOWN_CHECKPOINT_SIZES.get(path.name)
    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size < expected_size:
            raise ValueError(
                f"Checkpoint appears incomplete: {path} "
                f"(got {actual_size} bytes, expected {expected_size}). "
                "Wait for the download to finish or replace the file with a complete checkpoint."
            )
    return path
