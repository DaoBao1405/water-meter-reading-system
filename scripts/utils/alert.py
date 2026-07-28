from __future__ import annotations


def detect_alert(predicted_apluc: float, low_threshold: float, high_threshold: float):
    if predicted_apluc < low_threshold:
        return "LOW_PRESSURE", "Áp lực nước không đủ"
    if predicted_apluc > high_threshold:
        return "HIGH_PRESSURE", "Áp lực nước lớn"
    return "NORMAL", "Áp lực nước bình thường"
