from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class CvatAnnotation:
    name: str
    size: tuple[int, int]
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class PreparedDatasetSummary:
    source_archive: str
    total_images: int
    split_counts: dict[str, int]
    label: str
    output_dir: str


def parse_cvat_annotations(xml_path: str | Path, *, label: str = "prep") -> list[CvatAnnotation]:
    root = ET.parse(xml_path).getroot()
    annotations: list[CvatAnnotation] = []

    for image in root.findall("image"):
        polygons = [node for node in image.findall("polygon") if node.attrib.get("label") == label]
        if not polygons:
            continue

        polygon = polygons[0]
        points = []
        for pair in polygon.attrib["points"].split(";"):
            x_str, y_str = pair.split(",")
            points.append((float(x_str), float(y_str)))

        annotations.append(
            CvatAnnotation(
                name=image.attrib["name"],
                size=(int(float(image.attrib["width"])), int(float(image.attrib["height"]))),
                points=points,
            )
        )

    return annotations


def rasterize_polygon_mask(size: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255, outline=255)
    return np.asarray(mask, dtype=np.uint8)


def build_split_map(
    image_names: list[str],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    if not image_names:
        return {}
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    names = list(image_names)
    random.Random(seed).shuffle(names)
    n = len(names)
    train_count = max(1, math.floor(n * train_ratio))
    val_count = math.floor(n * val_ratio)
    if train_count + val_count > n:
        val_count = max(0, n - train_count)

    split_map: dict[str, str] = {}
    for idx, name in enumerate(names):
        if idx < train_count:
            split_map[name] = "train"
        elif idx < train_count + val_count:
            split_map[name] = "val"
        else:
            split_map[name] = "test"
    return split_map


def export_cvat_zip_to_segmentation_dataset(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    label: str = "prep",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    excluded_images: set[str] | None = None,
) -> PreparedDatasetSummary:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    with tempfile.TemporaryDirectory(prefix="tooth_cvat_") as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(temp_path)

        xml_path = temp_path / "annotations.xml"
        if not xml_path.exists():
            raise FileNotFoundError("annotations.xml not found in CVAT archive")

        annotations = parse_cvat_annotations(xml_path, label=label)
        if excluded_images:
            excluded = set(excluded_images)
            annotations = [item for item in annotations if item.name not in excluded]
        if not annotations:
            raise ValueError(f"No '{label}' polygons found in annotations.xml")

        split_map = build_split_map([item.name for item in annotations], train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

        for split in ("train", "val", "test"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "masks" / split).mkdir(parents=True, exist_ok=True)

        for annotation in annotations:
            split = split_map[annotation.name]
            source_image = next((path for path in (temp_path / "images").rglob(annotation.name) if path.is_file()), None)
            if source_image is None:
                raise FileNotFoundError(f"Image '{annotation.name}' referenced by annotations.xml was not found in archive")

            target_image = output_dir / "images" / split / annotation.name
            shutil.copy2(source_image, target_image)

            mask = rasterize_polygon_mask(annotation.size, annotation.points)
            target_mask = output_dir / "masks" / split / f"{Path(annotation.name).stem}.png"
            Image.fromarray(mask, mode="L").save(target_mask)

        split_counts = {split: sum(1 for value in split_map.values() if value == split) for split in ("train", "val", "test")}
        summary = PreparedDatasetSummary(
            source_archive=str(archive_path),
            total_images=len(annotations),
            split_counts=split_counts,
            label=label,
            output_dir=str(output_dir),
        )
        (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary
