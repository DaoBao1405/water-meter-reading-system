from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def x_center(self) -> float:
        x1, _, x2, _ = self.xyxy
        return (x1 + x2) / 2.0

    @property
    def y_center(self) -> float:
        _, y1, _, y2 = self.xyxy
        return (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class DigitDetection:
    digit: str
    role: str | None
    detection: Detection


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def parse_digit_detection(detection: Detection) -> DigitDetection | None:
    label = normalize_label(detection.label)
    if any(token in label for token in ("meter", "panel", "window", "counter", "register", "area")):
        return None

    if re.fullmatch(r"\d", label):
        return DigitDetection(label, None, detection)

    match = re.fullmatch(r"(?:digit|number|black|red|main|integer|decimal|fraction)_([0-9])", label)
    if not match:
        return None

    role = None
    if any(token in label for token in ("red", "decimal", "fraction")):
        role = "decimal"
    elif any(token in label for token in ("black", "main", "integer")):
        role = "main"

    return DigitDetection(match.group(1), role, detection)


def is_panel_detection(detection: Detection) -> bool:
    label = normalize_label(detection.label)
    return any(token in label for token in ("digit_panel", "number_panel", "reading_window", "panel", "counter", "register"))


def center_inside(inner: Detection, outer: Detection, margin_ratio: float = 0.04) -> bool:
    x1, y1, x2, y2 = outer.xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return (
        x1 - margin_x <= inner.x_center <= x2 + margin_x
        and y1 - margin_y <= inner.y_center <= y2 + margin_y
    )


def detections_from_ultralytics(result: Any) -> list[Detection]:
    names = result.names
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = str(names.get(class_id, class_id))
        confidence = float(box.conf.item())
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(Detection(label=label, class_id=class_id, confidence=confidence, xyxy=xyxy))
    return detections


def build_reading(
    detections: list[Detection],
    confidence_threshold: float = 0.25,
    panel_only: bool = True,
    decimal_digits: int = 0,
) -> dict[str, Any]:
    panels = [detection for detection in detections if is_panel_detection(detection)]
    selected_panel = max(panels, key=lambda item: item.area, default=None)

    digit_detections = [
        parsed
        for detection in detections
        if detection.confidence >= confidence_threshold
        for parsed in [parse_digit_detection(detection)]
        if parsed is not None
    ]

    if panel_only and selected_panel is not None:
        digit_detections = [
            digit for digit in digit_detections if center_inside(digit.detection, selected_panel)
        ]

    digit_detections = sorted(digit_detections, key=lambda item: (item.detection.x_center, item.detection.y_center))

    main_digits = [digit for digit in digit_detections if digit.role != "decimal"]
    decimal_digits_detected = [digit for digit in digit_detections if digit.role == "decimal"]

    if decimal_digits_detected:
        main_value = "".join(digit.digit for digit in main_digits)
        decimal_value = "".join(digit.digit for digit in decimal_digits_detected)
    else:
        raw_value = "".join(digit.digit for digit in digit_detections)
        if decimal_digits > 0 and len(raw_value) > decimal_digits:
            main_value = raw_value[:-decimal_digits]
            decimal_value = raw_value[-decimal_digits:]
        else:
            main_value = raw_value
            decimal_value = ""

    reading = f"{main_value}.{decimal_value}" if decimal_value else main_value
    confidences = [digit.detection.confidence for digit in digit_detections]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "reading": reading,
        "main_value": main_value,
        "decimal_value": decimal_value,
        "digit_count": len(digit_detections),
        "avg_digit_confidence": round(avg_confidence, 4),
        "panel_used": selected_panel is not None and panel_only,
        "digits": [
            {
                "digit": digit.digit,
                "role": digit.role,
                "label": digit.detection.label,
                "confidence": round(digit.detection.confidence, 4),
                "xyxy": [round(value, 2) for value in digit.detection.xyxy],
            }
            for digit in digit_detections
        ],
    }
