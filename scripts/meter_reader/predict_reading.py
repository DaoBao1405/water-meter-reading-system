from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .postprocess import build_reading, detections_from_ultralytics
except ImportError:
    from postprocess import build_reading, detections_from_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict water meter reading from image(s).")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLO weights, usually best.pt.")
    parser.add_argument("--source", required=True, help="Image, folder, video, webcam index, or stream URL.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--decimal-digits",
        type=int,
        default=0,
        help="If red/decimal classes are not used, split the last N digits as decimal digits.",
    )
    parser.add_argument(
        "--all-digits",
        action="store_true",
        help="Read digits from the whole image instead of only inside the detected digit_panel.",
    )
    parser.add_argument("--save-vis", type=Path, default=None, help="Optional directory for annotated images.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON result file.")
    return parser.parse_args()


def draw_visualization(image_path: Path, detections, reading: str, output_path: Path) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        label = f"{detection.label} {detection.confidence:.2f}"
        color = (0, 180, 255)
        if "panel" in detection.label.lower():
            color = (255, 0, 255)
        elif any(char.isdigit() for char in detection.label):
            color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(image, f"reading: {reading}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.predict(source=args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)

    output_rows = []
    for result in results:
        detections = detections_from_ultralytics(result)
        reading = build_reading(
            detections=detections,
            confidence_threshold=args.conf,
            panel_only=not args.all_digits,
            decimal_digits=args.decimal_digits,
        )
        image_path = Path(result.path)
        row = {
            "source": str(image_path),
            **reading,
        }
        output_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if args.save_vis is not None:
            output_path = args.save_vis / f"{image_path.stem}_meter_reading{image_path.suffix}"
            draw_visualization(image_path, detections, reading["reading"], output_path)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
