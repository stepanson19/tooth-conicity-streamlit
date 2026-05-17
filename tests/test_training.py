from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from tooth_service.training import (
    PrepSegmentationDataset,
    TinyUNet,
    compute_segmentation_scores,
    evaluate_segmentation_batches,
    keep_largest_connected_component,
)


def test_prep_segmentation_dataset_loads_aligned_image_mask_pair(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    Image.new("RGB", (20, 10), color=(20, 40, 60)).save(images_dir / "sample.jpg")
    Image.new("L", (20, 10), color=255).save(masks_dir / "sample.png")

    dataset = PrepSegmentationDataset(images_dir, masks_dir, image_size=32)
    image, mask, name = dataset[0]

    assert name == "sample"
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert float(mask.max()) == 1.0


def test_tiny_unet_returns_single_channel_segmentation_logits():
    model = TinyUNet(in_channels=3, base_channels=8)
    x = torch.randn(2, 3, 64, 64)

    y = model(x)

    assert y.shape == (2, 1, 64, 64)


def test_keep_largest_connected_component_removes_small_islands():
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask[0, 0, 1:5, 1:5] = 1.0
    mask[0, 0, 6, 6] = 1.0
    mask[0, 0, 0, 7] = 1.0

    filtered = keep_largest_connected_component(mask)

    assert filtered.sum().item() == 16.0
    assert filtered[0, 0, 6, 6].item() == 0.0
    assert filtered[0, 0, 0, 7].item() == 0.0


def test_compute_segmentation_scores_reports_accuracy_dice_and_iou():
    logits = torch.tensor(
        [
            [[
                [10.0, -10.0, -10.0],
                [10.0, -10.0, -10.0],
            ]],
            [[
                [-10.0, -10.0, -10.0],
                [-10.0, -10.0, -10.0],
            ]],
        ]
    )
    targets = torch.tensor(
        [
            [[
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
            [[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
        ]
    )

    scores = compute_segmentation_scores(logits, targets)

    assert scores.accuracy == pytest.approx((4 / 6 + 1.0) / 2)
    assert scores.dice == pytest.approx((0.5 + 1.0) / 2)
    assert scores.iou == pytest.approx((1 / 3 + 1.0) / 2)


def test_evaluate_segmentation_batches_reports_per_image_metrics():
    logits = torch.tensor(
        [
            [[
                [10.0, -10.0],
                [-10.0, -10.0],
            ]],
            [[
                [10.0, 10.0],
                [-10.0, -10.0],
            ]],
        ]
    )
    targets = torch.tensor(
        [
            [[
                [1.0, 0.0],
                [0.0, 0.0],
            ]],
            [[
                [1.0, 0.0],
                [1.0, 0.0],
            ]],
        ]
    )

    report = evaluate_segmentation_batches([(logits, targets, ["perfect", "mixed"])])

    assert report["average"]["accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert report["average"]["iou"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert report["images"][0]["image"] == "perfect"
    assert report["images"][0]["accuracy"] == pytest.approx(1.0)
    assert report["images"][0]["iou"] == pytest.approx(1.0)
    assert report["images"][1]["image"] == "mixed"
    assert report["images"][1]["accuracy"] == pytest.approx(0.5)
    assert report["images"][1]["iou"] == pytest.approx(1 / 3)
