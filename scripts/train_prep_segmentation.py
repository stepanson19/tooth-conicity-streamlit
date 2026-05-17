from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tooth_service.cvat_dataset import export_cvat_zip_to_segmentation_dataset
from tooth_service.training import (
    PrepSegmentationDataset,
    evaluate_model,
    evaluate_model_by_image,
    save_training_result,
    train_segmentation_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small prepared-tooth segmentation baseline from a CVAT ZIP export.")
    parser.add_argument("--cvat-zip", required=True, help="Path to the CVAT ZIP archive.")
    parser.add_argument("--prepared-dir", default="artifacts/prepared_dataset", help="Where to export images/masks.")
    parser.add_argument("--run-dir", default="artifacts/runs/latest", help="Where to save training outputs.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-image", action="append", default=[], help="Image filename to exclude from the dataset. Can be passed multiple times.")
    parser.add_argument("--force", action="store_true", help="Delete previous prepared-dir/run-dir contents before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    prepared_dir = Path(args.prepared_dir)
    run_dir = Path(args.run_dir)
    if args.force:
        shutil.rmtree(prepared_dir, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = export_cvat_zip_to_segmentation_dataset(
        args.cvat_zip,
        prepared_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        excluded_images=set(args.exclude_image),
    )

    train_dataset = PrepSegmentationDataset(prepared_dir / "images" / "train", prepared_dir / "masks" / "train", image_size=args.image_size)
    val_dataset = PrepSegmentationDataset(prepared_dir / "images" / "val", prepared_dir / "masks" / "val", image_size=args.image_size)
    test_dataset = PrepSegmentationDataset(prepared_dir / "images" / "test", prepared_dir / "masks" / "test", image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_result = train_segmentation_model(
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics_largest = evaluate_model(model, test_loader, device, largest_component_only=True)
    test_metrics_by_image = evaluate_model_by_image(model, test_loader, device)
    test_metrics_largest_by_image = evaluate_model_by_image(model, test_loader, device, largest_component_only=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    save_training_result(training_result, run_dir / "training_metrics.json")
    (run_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(asdict(test_metrics), indent=2), encoding="utf-8")
    (run_dir / "test_metrics_largest_component.json").write_text(json.dumps(asdict(test_metrics_largest), indent=2), encoding="utf-8")
    (run_dir / "test_metrics_by_image.json").write_text(json.dumps(test_metrics_by_image, indent=2), encoding="utf-8")
    (run_dir / "test_metrics_largest_component_by_image.json").write_text(json.dumps(test_metrics_largest_by_image, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "prepared_dir": str(prepared_dir),
            "run_dir": str(run_dir),
            "device": str(device),
            "total_images": summary.total_images,
            "split_counts": summary.split_counts,
            "best_epoch": training_result.best_epoch,
            "best_val_iou": training_result.best_val_iou,
            "test_metrics": asdict(test_metrics),
            "test_metrics_largest_component": asdict(test_metrics_largest),
            "test_metrics_by_image": test_metrics_by_image["average"],
            "test_metrics_largest_component_by_image": test_metrics_largest_by_image["average"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
