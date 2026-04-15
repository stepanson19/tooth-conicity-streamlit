from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from tooth_service.cvat_dataset import (
    build_split_map,
    export_cvat_zip_to_segmentation_dataset,
    parse_cvat_annotations,
    rasterize_polygon_mask,
)


def test_rasterize_polygon_mask_fills_polygon_area():
    mask = rasterize_polygon_mask((12, 12), [(2, 2), (9, 2), (9, 9), (2, 9)])

    assert mask.shape == (12, 12)
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 255
    assert mask[0, 0] == 0


def test_parse_cvat_annotations_ignores_background_boxes(tmp_path: Path):
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="a.jpg" width="10" height="8">
    <box label="background" xtl="0" ytl="0" xbr="10" ybr="2" />
    <polygon label="prep" points="2,2;7,2;7,6;2,6" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    annotations = parse_cvat_annotations(xml_path)

    assert len(annotations) == 1
    assert annotations[0].name == "a.jpg"
    assert annotations[0].size == (10, 8)
    assert annotations[0].points == [(2.0, 2.0), (7.0, 2.0), (7.0, 6.0), (2.0, 6.0)]


def test_build_split_map_creates_deterministic_train_val_test_split():
    names = [f"IMG_{idx:04d}.JPG" for idx in range(20)]

    split_map = build_split_map(names, train_ratio=0.7, val_ratio=0.15, seed=7)

    assert list(split_map.values()).count("train") == 14
    assert list(split_map.values()).count("val") == 3
    assert list(split_map.values()).count("test") == 3
    assert split_map == build_split_map(names, train_ratio=0.7, val_ratio=0.15, seed=7)


def test_export_cvat_zip_to_segmentation_dataset_writes_images_masks_and_metadata(tmp_path: Path):
    archive_path = tmp_path / "toy.zip"
    source_dir = tmp_path / "src"
    images_dir = source_dir / "images" / "Train"
    images_dir.mkdir(parents=True)

    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :] = np.array([10, 20, 30], dtype=np.uint8)
    Image.fromarray(image).save(images_dir / "a.jpg")
    Image.fromarray(image).save(images_dir / "b.jpg")

    (source_dir / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="a.jpg" width="10" height="8">
    <polygon label="prep" points="2,2;7,2;7,6;2,6" />
    <box label="background" xtl="0" ytl="0" xbr="10" ybr="2" />
  </image>
  <image id="1" name="b.jpg" width="10" height="8">
    <polygon label="prep" points="1,1;4,1;4,5;1,5" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(source_dir / "annotations.xml", "annotations.xml")
        zf.write(images_dir / "a.jpg", "images/Train/a.jpg")
        zf.write(images_dir / "b.jpg", "images/Train/b.jpg")

    output_dir = tmp_path / "prepared"
    summary = export_cvat_zip_to_segmentation_dataset(
        archive_path,
        output_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=3,
    )

    assert summary.total_images == 2
    assert summary.split_counts == {"train": 1, "val": 0, "test": 1}
    train_masks = list((output_dir / "masks" / "train").glob("*.png"))
    test_masks = list((output_dir / "masks" / "test").glob("*.png"))
    assert len(train_masks) == 1
    assert len(test_masks) == 1

    exported_mask = np.array(Image.open(train_masks[0]))
    assert exported_mask.ndim == 2
    assert set(np.unique(exported_mask)).issubset({0, 255})
    assert (output_dir / "dataset_summary.json").exists()


def test_export_cvat_zip_to_segmentation_dataset_supports_flat_images_dir_and_exclusions(tmp_path: Path):
    archive_path = tmp_path / "flat.zip"
    source_dir = tmp_path / "src_flat"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)

    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :] = np.array([10, 20, 30], dtype=np.uint8)
    Image.fromarray(image).save(images_dir / "keep.jpg")
    Image.fromarray(image).save(images_dir / "drop.jpg")

    (source_dir / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="keep.jpg" width="10" height="8">
    <polygon label="prep" points="2,2;7,2;7,6;2,6" />
  </image>
  <image id="1" name="drop.jpg" width="10" height="8">
    <polygon label="prep" points="1,1;4,1;4,5;1,5" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(source_dir / "annotations.xml", "annotations.xml")
        zf.write(images_dir / "keep.jpg", "images/keep.jpg")
        zf.write(images_dir / "drop.jpg", "images/drop.jpg")

    output_dir = tmp_path / "prepared_flat"
    summary = export_cvat_zip_to_segmentation_dataset(
        archive_path,
        output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=3,
        excluded_images={"drop.jpg"},
    )

    assert summary.total_images == 1
    assert summary.split_counts == {"train": 1, "val": 0, "test": 0}
    assert (output_dir / "images" / "train" / "keep.jpg").exists()
    assert not (output_dir / "images" / "train" / "drop.jpg").exists()
