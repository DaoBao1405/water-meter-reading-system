from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


def _intersect_and_to_crop(
    box: list[float], crop_box: list[int]
) -> list[float] | None:
    """Intersect a box in full-image coordinates with a crop and localize it."""
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = crop_box

    ix1, iy1 = max(bx1, cx1), max(by1, cy1)
    ix2, iy2 = min(bx2, cx2), min(by2, cy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return [ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1]


def _center_inside_box(x: float, y: float, box: list[float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _sort_digits_toward_liter(
    digits: list[dict[str, Any]], liter_boxes_in_crop: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Order digits in the direction that approaches the liter-area box."""
    if len(digits) <= 1:
        return digits, "unknown"

    x_centers = np.array([item["x_center"] for item in digits])
    y_centers = np.array([item["y_center"] for item in digits])
    is_vertical = float(np.ptp(y_centers)) > float(np.ptp(x_centers))

    if liter_boxes_in_crop:
        best_liter = max(
            liter_boxes_in_crop, key=lambda item: item["confidence"]
        )
        lx1, ly1, lx2, ly2 = best_liter["xyxy"]
        liter_x = (lx1 + lx2) / 2
        liter_y = (ly1 + ly2) / 2
    else:
        liter_x = liter_y = None

    if is_vertical:
        digit_middle = float(np.median(y_centers))
        reverse = liter_y is not None and liter_y < digit_middle
        return (
            sorted(digits, key=lambda item: item["y_center"], reverse=reverse),
            "bottom_to_top" if reverse else "top_to_bottom",
        )

    digit_middle = float(np.median(x_centers))
    reverse = liter_x is not None and liter_x < digit_middle
    return (
        sorted(digits, key=lambda item: item["x_center"], reverse=reverse),
        "right_to_left" if reverse else "left_to_right",
    )


def image_from_bytes(raw_image: bytes) -> Image.Image:
    """Decode an uploaded image and normalize EXIF orientation."""
    try:
        with Image.open(BytesIO(raw_image)) as opened_image:
            return ImageOps.exif_transpose(opened_image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("Tệp tải lên không phải ảnh hợp lệ.") from exc


@dataclass
class MeterReader:
    counter_model: YOLO
    digit_model: YOLO
    device: str | int

    @classmethod
    def from_weights(
        cls,
        counter_weights: Path,
        digit_weights: Path,
        device: str | int,
    ) -> "MeterReader":
        if not counter_weights.is_file():
            raise FileNotFoundError(f"Không tìm thấy counter weights: {counter_weights}")
        if not digit_weights.is_file():
            raise FileNotFoundError(f"Không tìm thấy digit weights: {digit_weights}")

        return cls(
            counter_model=YOLO(str(counter_weights)),
            digit_model=YOLO(str(digit_weights)),
            device=device,
        )

    def predict(
        self,
        full_image: Image.Image,
        region_conf: float = 0.25,
        digit_conf: float = 0.25,
        rotate_degrees: int = 0,
        expected_digit_count: int = 4,
    ) -> dict[str, Any]:
        if rotate_degrees:
            full_image = full_image.rotate(rotate_degrees, expand=True)

        region_result = self.counter_model.predict(
            source=np.array(full_image),
            conf=region_conf,
            iou=0.30,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        counter_boxes: list[dict[str, Any]] = []
        liter_boxes: list[dict[str, Any]] = []

        for box in region_result.boxes:
            class_name = str(region_result.names[int(box.cls[0])])
            item = {
                "xyxy": [float(value) for value in box.xyxy[0].cpu().tolist()],
                "confidence": round(float(box.conf[0]), 3),
            }
            if class_name == "counter":
                counter_boxes.append(item)
            elif class_name == "liter":
                liter_boxes.append(item)

        base = {"image_size": {"width": full_image.width, "height": full_image.height}}
        if not counter_boxes:
            return {**base, "status": "counter_not_found"}

        best_counter = max(counter_boxes, key=lambda item: item["confidence"])
        x1, y1, x2, y2 = map(int, best_counter["xyxy"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_image.width, x2), min(full_image.height, y2)

        if x2 <= x1 or y2 <= y1:
            return {**base, "status": "invalid_counter_box"}

        counter_box = [x1, y1, x2, y2]
        counter_crop = full_image.crop(tuple(counter_box))
        liter_boxes_in_crop = []

        for liter_item in liter_boxes:
            local_box = _intersect_and_to_crop(liter_item["xyxy"], counter_box)
            if local_box is not None:
                liter_boxes_in_crop.append(
                    {"xyxy": local_box, "confidence": liter_item["confidence"]}
                )

        digit_input = counter_crop.copy()
        mask_draw = ImageDraw.Draw(digit_input)
        for liter_item in liter_boxes_in_crop:
            lx1, ly1, lx2, ly2 = liter_item["xyxy"]
            mask_draw.rectangle(
                (
                    max(0, int(lx1) - 4),
                    max(0, int(ly1) - 4),
                    min(digit_input.width, int(lx2) + 4),
                    min(digit_input.height, int(ly2) + 4),
                ),
                fill=(127, 127, 127),
            )

        digit_result = self.digit_model.predict(
            source=np.array(digit_input),
            conf=digit_conf,
            iou=0.40,
            agnostic_nms=True,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        digits: list[dict[str, Any]] = []
        ignored_liter_digits: list[dict[str, Any]] = []
        for box in digit_result.boxes:
            dx1, dy1, dx2, dy2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
            x_center, y_center = (dx1 + dx2) / 2, (dy1 + dy2) / 2
            item = {
                "digit": str(digit_result.names[int(box.cls[0])]),
                "xyxy": [dx1, dy1, dx2, dy2],
                "x_center": x_center,
                "y_center": y_center,
                "confidence": round(float(box.conf[0]), 3),
            }
            if any(
                _center_inside_box(x_center, y_center, liter["xyxy"])
                for liter in liter_boxes_in_crop
            ):
                ignored_liter_digits.append(item)
            else:
                digits.append(item)

        digits, reading_direction = _sort_digits_toward_liter(
            digits, liter_boxes_in_crop
        )
        detected_reading = "".join(item["digit"] for item in digits)
        average_confidence = (
            round(sum(item["confidence"] for item in digits) / len(digits), 3)
            if digits
            else None
        )

        # A partial reading is unsafe to present as a meter value. Keep its
        # boxes for review, but require the configured number of digits.
        if not digits:
            status = "digits_not_found"
            reading = None
        elif len(digits) < expected_digit_count:
            status = "incomplete_digits"
            reading = None
        else:
            status = "ok"
            reading = detected_reading

        return {
            **base,
            "status": status,
            "reading": reading,
            "reading_direction": reading_direction,
            "counter_box": counter_box,
            "counter_confidence": best_counter["confidence"],
            "liter_boxes": liter_boxes,
            "liter_boxes_in_crop": liter_boxes_in_crop,
            "digits": digits,
            "ignored_liter_digits": ignored_liter_digits,
            "average_digit_confidence": average_confidence,
        }

    @staticmethod
    def annotate(full_image: Image.Image, result: dict[str, Any]) -> bytes:
        """Return a PNG with the accepted detections drawn on it."""
        image = full_image.copy()
        draw = ImageDraw.Draw(image)

        if result.get("counter_box"):
            draw.rectangle(result["counter_box"], outline="lime", width=4)
            draw.text(
                (result["counter_box"][0], max(0, result["counter_box"][1] - 18)),
                f"counter {result['counter_confidence']:.2f}",
                fill="lime",
            )

        for liter in result.get("liter_boxes", []):
            draw.rectangle(liter["xyxy"], outline="orange", width=4)

        if result.get("counter_box"):
            offset_x, offset_y = result["counter_box"][:2]
            for digit in result.get("digits", []):
                x1, y1, x2, y2 = digit["xyxy"]
                box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
                draw.rectangle(box, outline="lime", width=3)
                draw.text((box[0], max(0, box[1] - 16)), digit["digit"], fill="lime")

        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
