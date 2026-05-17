from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader, Dataset


class PrepSegmentationDataset(Dataset):
    def __init__(self, images_dir: str | Path, masks_dir: str | Path, *, image_size: int = 256):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.image_paths = sorted(self.images_dir.glob("*"))
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        stem = image_path.stem
        mask_path = self.masks_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image '{image_path.name}'")

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)[None, ...]
        image_tensor = torch.from_numpy(image_array)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor, stem


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, *, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.enc3 = _ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


@dataclass(frozen=True)
class SegmentationScores:
    accuracy: float
    dice: float
    iou: float


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    dice: float
    iou: float


@dataclass(frozen=True)
class TrainingResult:
    train: list[EpochMetrics]
    val: list[EpochMetrics]
    best_val_iou: float
    best_epoch: int


def _compute_binary_segmentation_scores(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> SegmentationScores:
    pred_mask = preds.bool()
    target_mask = targets >= 0.5
    true_positive = (pred_mask & target_mask).sum(dim=(1, 2, 3)).float()
    true_negative = (~pred_mask & ~target_mask).sum(dim=(1, 2, 3)).float()
    false_positive = (pred_mask & ~target_mask).sum(dim=(1, 2, 3)).float()
    false_negative = (~pred_mask & target_mask).sum(dim=(1, 2, 3)).float()

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = torch.where(total > 0, (true_positive + true_negative) / total, torch.ones_like(total))

    iou_denominator = true_positive + false_positive + false_negative
    iou = torch.where(iou_denominator > 0, true_positive / iou_denominator, torch.ones_like(iou_denominator))

    dice_denominator = 2.0 * true_positive + false_positive + false_negative
    dice = torch.where(dice_denominator > 0, 2.0 * true_positive / dice_denominator, torch.ones_like(dice_denominator))

    return SegmentationScores(
        accuracy=float(accuracy.mean().item()),
        dice=float(dice.mean().item()),
        iou=float(iou.mean().item()),
    )


def compute_segmentation_scores(logits: torch.Tensor, targets: torch.Tensor) -> SegmentationScores:
    preds = torch.sigmoid(logits) >= 0.5
    return _compute_binary_segmentation_scores(preds, targets)


def _as_score_dict(scores: SegmentationScores) -> dict[str, float]:
    return {
        "accuracy": scores.accuracy,
        "dice": scores.dice,
        "iou": scores.iou,
    }


def evaluate_binary_segmentation_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
) -> dict[str, object]:
    rows: list[dict[str, float | str]] = []
    totals = SegmentationScores(accuracy=0.0, dice=0.0, iou=0.0)
    count = 0
    for preds, targets, names in batches:
        if preds.shape[0] != targets.shape[0] or preds.shape[0] != len(names):
            raise ValueError("Predictions, targets, and names must have the same batch size")
        for index, name in enumerate(names):
            scores = _compute_binary_segmentation_scores(preds[index : index + 1], targets[index : index + 1])
            rows.append({"image": str(name), **_as_score_dict(scores)})
            totals = SegmentationScores(
                accuracy=totals.accuracy + scores.accuracy,
                dice=totals.dice + scores.dice,
                iou=totals.iou + scores.iou,
            )
            count += 1

    if count == 0:
        average = SegmentationScores(accuracy=0.0, dice=0.0, iou=0.0)
    else:
        average = SegmentationScores(
            accuracy=totals.accuracy / count,
            dice=totals.dice / count,
            iou=totals.iou / count,
        )
    return {"average": _as_score_dict(average), "images": rows}


def evaluate_segmentation_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
) -> dict[str, object]:
    binary_batches = (
        (torch.sigmoid(logits) >= 0.5, targets, names)
        for logits, targets, names in batches
    )
    return evaluate_binary_segmentation_batches(binary_batches)


def compute_batch_scores(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    del eps
    scores = compute_segmentation_scores(logits, targets)
    return scores.dice, scores.iou


def keep_largest_connected_component(binary_masks: torch.Tensor) -> torch.Tensor:
    masks = binary_masks.detach().cpu().numpy()
    filtered = np.zeros_like(masks, dtype=np.float32)
    for batch_idx in range(masks.shape[0]):
        mask = masks[batch_idx, 0].astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            filtered[batch_idx, 0] = mask
            continue
        best_label = 1
        best_area = 0
        for label in range(1, num_labels):
            area = int((labels == label).sum())
            if area > best_area:
                best_area = area
                best_label = label
        filtered[batch_idx, 0] = (labels == best_label).astype(np.float32)
    return torch.from_numpy(filtered)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    largest_component_only: bool = False,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if largest_component_only:
                preds = keep_largest_connected_component((torch.sigmoid(logits) >= 0.5).float()).to(masks.device)
                scores = _compute_binary_segmentation_scores(preds, masks)
            else:
                scores = compute_segmentation_scores(logits, masks)
            batch_size = int(masks.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_accuracy += scores.accuracy * batch_size
            total_dice += scores.dice * batch_size
            total_iou += scores.iou * batch_size
            total_images += batch_size
    if total_images == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0, dice=0.0, iou=0.0)
    return EpochMetrics(
        loss=total_loss / total_images,
        accuracy=total_accuracy / total_images,
        dice=total_dice / total_images,
        iou=total_iou / total_images,
    )


def evaluate_model_by_image(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    largest_component_only: bool = False,
) -> dict[str, object]:
    model.eval()
    batches = []
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            if largest_component_only:
                preds = keep_largest_connected_component((torch.sigmoid(logits) >= 0.5).float()).to(masks.device)
                batches.append((preds, masks, names))
            else:
                batches.append((torch.sigmoid(logits) >= 0.5, masks, names))
    return evaluate_binary_segmentation_batches(batches)


def train_segmentation_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
) -> tuple[nn.Module, TrainingResult]:
    model = TinyUNet(base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_history: list[EpochMetrics] = []
    val_history: list[EpochMetrics] = []
    best_val_iou = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_dice = 0.0
        train_iou = 0.0
        image_count = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            scores = compute_segmentation_scores(logits.detach(), masks)
            batch_size = int(masks.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_accuracy += scores.accuracy * batch_size
            train_dice += scores.dice * batch_size
            train_iou += scores.iou * batch_size
            image_count += batch_size

        train_metrics = EpochMetrics(
            loss=train_loss / max(1, image_count),
            accuracy=train_accuracy / max(1, image_count),
            dice=train_dice / max(1, image_count),
            iou=train_iou / max(1, image_count),
        )
        val_metrics = evaluate_model(model, val_loader, device)
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if val_metrics.iou > best_val_iou:
            best_val_iou = val_metrics.iou
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainingResult(
        train=train_history,
        val=val_history,
        best_val_iou=best_val_iou,
        best_epoch=best_epoch,
    )


def save_training_result(training_result: TrainingResult, output_path: str | Path) -> None:
    payload = {
        "train": [asdict(item) for item in training_result.train],
        "val": [asdict(item) for item in training_result.val],
        "best_val_iou": training_result.best_val_iou,
        "best_epoch": training_result.best_epoch,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
