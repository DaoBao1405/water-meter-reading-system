from __future__ import annotations

import unittest

from PIL import Image

from app.pipeline import MeterReader, _sort_digits_toward_liter


class FakeArray:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, class_id, xyxy, confidence=0.9):
        self.cls = [class_id]
        self.xyxy = [FakeArray(xyxy)]
        self.conf = [confidence]


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.sources = []

    def predict(self, **kwargs):
        self.sources.append(kwargs["source"])
        return [self.result]


def digit_boxes(count):
    return [
        FakeBox(index % 10, [index * 10, 10, index * 10 + 8, 30])
        for index in range(count)
    ]


class PipelineV2Tests(unittest.TestCase):
    def make_reader(self, digit_count):
        counter_result = FakeResult(
            {0: "counter", 1: "liter"},
            [FakeBox(0, [25, 25, 75, 75])],
        )
        digit_result = FakeResult(
            {index: str(index) for index in range(10)},
            digit_boxes(digit_count),
        )
        return MeterReader(FakeModel(counter_result), FakeModel(digit_result), "cpu")

    def test_counter_crop_uses_four_percent_padding(self):
        reader = self.make_reader(4)

        result = reader.predict(Image.new("RGB", (100, 100)), expected_digit_count=4)

        self.assertEqual(result["counter_box"], [23, 23, 77, 77])
        self.assertEqual(reader.digit_model.sources[0].shape[:2], (54, 54))

    def test_only_exact_digit_count_is_accepted(self):
        for count, expected_status in [
            (0, "digits_not_found"),
            (3, "unexpected_digit_count"),
            (4, "ok"),
            (5, "unexpected_digit_count"),
        ]:
            with self.subTest(count=count):
                result = self.make_reader(count).predict(
                    Image.new("RGB", (100, 100)), expected_digit_count=4
                )
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["reading"] is not None, count == 4)

    def test_pca_orders_diagonal_digits_toward_liter(self):
        digits = [
            {"digit": "3", "x_center": 30.0, "y_center": 30.0},
            {"digit": "1", "x_center": 10.0, "y_center": 10.0},
            {"digit": "2", "x_center": 20.0, "y_center": 20.0},
        ]
        liter = [{"xyxy": [38.0, 38.0, 42.0, 42.0], "confidence": 0.9}]

        ordered, direction = _sort_digits_toward_liter(digits, liter)

        self.assertEqual("".join(item["digit"] for item in ordered), "123")
        self.assertTrue(direction.startswith("pca_axis_"))


if __name__ == "__main__":
    unittest.main()
