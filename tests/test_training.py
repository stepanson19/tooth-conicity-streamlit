from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from tooth_service.training import PrepSegmentationDataset, TinyUNet, keep_largest_connected_component


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
