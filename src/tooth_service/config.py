from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def resolve_checkpoint_path(path_like: PathLike) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve()


def ensure_checkpoint_exists(path_like: PathLike) -> Path:
    path = resolve_checkpoint_path(path_like)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
