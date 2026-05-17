from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tooth_service.training import PrepSegmentationDataset, evaluate_model, evaluate_model_by_image
from tooth_service.trained_inference import load_trained_prep_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained prepared-tooth segmentation model.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model.pt file.")
    parser.add_argument("--prepared-dir", required=True, help="Prepared dataset directory with images/ and masks/ splits.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional JSON output path for the full metrics report.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--largest-component-only", action="store_true", help="Evaluate after keeping only the largest predicted component.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    dataset = PrepSegmentationDataset(
        prepared_dir / "images" / args.split,
        prepared_dir / "masks" / args.split,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_prep_model(args.model_path, device=str(device), base_channels=args.base_channels)
    average_metrics = evaluate_model(model, loader, device, largest_component_only=args.largest_component_only)
    by_image = evaluate_model_by_image(model, loader, device, largest_component_only=args.largest_component_only)

    report = {
        "model_path": str(args.model_path),
        "prepared_dir": str(prepared_dir),
        "split": args.split,
        "device": str(device),
        "largest_component_only": args.largest_component_only,
        "average": asdict(average_metrics),
        "by_image": by_image["images"],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
