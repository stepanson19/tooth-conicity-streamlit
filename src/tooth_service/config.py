from __future__ import annotations

import os
from pathlib import Path
from typing import Union
from urllib.request import urlopen

from .constants import DEFAULT_SAM_MODEL_TYPE, SAM_MODEL_TYPES


PathLike = Union[str, Path]

KNOWN_SAM_CHECKPOINTS = {
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "size": 2564550879,
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "size": 1249524607,
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "size": 375042383,
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
}
KNOWN_CHECKPOINT_SIZES = {item["filename"]: item["size"] for item in KNOWN_SAM_CHECKPOINTS.values()}
KNOWN_CHECKPOINT_URLS = {item["filename"]: item["url"] for item in KNOWN_SAM_CHECKPOINTS.values()}
AUTO_DOWNLOAD_CHECKPOINT_ENV = "TOOTH_SERVICE_AUTO_DOWNLOAD_CHECKPOINT"
DEFAULT_MODEL_TYPE_ENV = "TOOTH_SERVICE_DEFAULT_SAM_MODEL_TYPE"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoints"


def resolve_model_type(model_type: str | None = None) -> str:
    value = model_type or os.getenv(DEFAULT_MODEL_TYPE_ENV, DEFAULT_SAM_MODEL_TYPE)
    if value not in SAM_MODEL_TYPES:
        supported = ", ".join(SAM_MODEL_TYPES)
        raise ValueError(f"Unsupported SAM model type: {value!r}. Supported: {supported}")
    return value


def default_checkpoint_filename(model_type: str | None = None) -> str:
    return KNOWN_SAM_CHECKPOINTS[resolve_model_type(model_type)]["filename"]


def default_checkpoint_path(root_dir: PathLike, model_type: str | None = None) -> Path:
    root = Path(root_dir).expanduser().resolve()
    return root / DEFAULT_CHECKPOINT_DIRNAME / default_checkpoint_filename(model_type)


def resolve_checkpoint_path(path_like: PathLike) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve()


def _auto_download_enabled() -> bool:
    return os.getenv(AUTO_DOWNLOAD_CHECKPOINT_ENV, "").strip() == "1"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temp_path.replace(destination)


def _maybe_download_known_checkpoint(path: Path) -> None:
    url = KNOWN_CHECKPOINT_URLS.get(path.name)
    if url is None or not _auto_download_enabled():
        return
    _download_file(url, path)


def ensure_checkpoint_exists(path_like: PathLike) -> Path:
    path = resolve_checkpoint_path(path_like)
    expected_size = KNOWN_CHECKPOINT_SIZES.get(path.name)
    if (
        expected_size is not None
        and _auto_download_enabled()
        and (not path.exists() or not path.is_file() or path.stat().st_size < expected_size)
    ):
        _maybe_download_known_checkpoint(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size < expected_size:
            raise ValueError(
                f"Checkpoint appears incomplete: {path} "
                f"(got {actual_size} bytes, expected {expected_size}). "
                "Wait for the download to finish or replace the file with a complete checkpoint."
            )
    return path
