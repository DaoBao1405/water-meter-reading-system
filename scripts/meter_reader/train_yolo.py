from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for water meter digit detection.")
    parser.add_argument("--data", type=Path, default=Path("data/meter_digits/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO checkpoint, e.g. yolo11n.pt or yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Use '0' for first GPU, 'cpu' for CPU, or omit for auto.")
    parser.add_argument("--project", default="runs/meter_reader")
    parser.add_argument("--name", default="digits")
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
