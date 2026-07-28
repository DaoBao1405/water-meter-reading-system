import csv
import io
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ZIP_PATH = Path(r"D:\AIDOCSAI_5_2026.zip")
OUT_DIR = Path(r"D:\AIDOCSAI_yolo")
RANDOM_SEED = 42

CLASSES = [
    "loai_1_vanh_cam_do",
    "loai_2_than_den",
    "loai_3_nap_trang_vang",
    "loai_4_can_canh_mat_so",
    "loai_5_khac_khong_ro",
]


def image_entries(archive):
    suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    return [entry for entry in archive.infolist() if entry.filename.lower().endswith(suffixes)]


def safe_name(zip_name):
    return Path(zip_name).name


def load_image_from_zip(archive, entry):
    with archive.open(entry) as file_obj:
        image = Image.open(file_obj)
        image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image)


def load_bytes_and_image(archive, entry):
    data = archive.read(entry)
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return data, np.array(image)


def resize_for_detection(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    if scale == 1.0:
        return rgb, 1.0
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def detect_meter_bbox(rgb):
    small, scale = resize_for_detection(rgb)
    h, w = small.shape[:2]
    min_dim = min(h, w)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Fast foreground estimate: meters are usually centered and differ from
    # the pipe/ground background by saturation, brightness, or edge density.
    mask = (((sat > 45) & (val > 45)) | ((val > 115) & (sat < 80))).astype(np.uint8) * 255
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_dim * min_dim * 0.035:
            continue
        ratio = bw / max(bh, 1)
        if 0.35 <= ratio <= 2.5:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            dist = np.hypot(cx - w / 2.0, cy - h / 2.0) / max(min_dim, 1)
            candidates.append((area - dist * min_dim * min_dim * 0.45, x, y, bw, bh))
    if candidates:
        _, x, y, bw, bh = max(candidates)
        pad = int(max(bw, bh) * 0.16)
        box = (x - pad, y - pad, x + bw + pad, y + bh + pad)
    else:
        side = int(min_dim * 0.82)
        box = ((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)

    x1, y1, x2, y2 = clamp_bbox(*box, w, h)
    inv = 1.0 / scale
    oh, ow = rgb.shape[:2]
    return clamp_bbox(x1 * inv, y1 * inv, x2 * inv, y2 * inv, ow, oh)


def classify_meter(rgb, bbox):
    x1, y1, x2, y2 = bbox
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 4

    h, w = rgb.shape[:2]
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
    if box_area_ratio > 0.62:
        return 3

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    orange = (((hue <= 15) | (hue >= 170)) & (sat > 70) & (val > 70)).mean()
    dark = (val < 75).mean()
    beige = ((hue >= 14) & (hue <= 38) & (sat > 25) & (sat < 145) & (val > 95)).mean()
    white_yellow = (((sat < 45) & (val > 145)) | (beige > 0)).mean()

    if orange > 0.055:
        return 0
    if dark > 0.43:
        return 1
    if white_yellow > 0.58 or beige > 0.10:
        return 2
    return 4


def yolo_line(class_id, bbox, width, height):
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def split_name(index, total):
    if index < int(total * 0.80):
        return "train"
    if index < int(total * 0.90):
        return "val"
    return "test"


def write_data_yaml():
    text = [
        "path: D:/AIDOCSAI_yolo",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    text.extend([f"  {idx}: {name}" for idx, name in enumerate(CLASSES)])
    (OUT_DIR / "data.yaml").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        entries = image_entries(archive)
        random.shuffle(entries)
        total = len(entries)

        summary_path = OUT_DIR / "labels_summary.csv"
        write_header = not summary_path.exists() or summary_path.stat().st_size == 0
        with summary_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "source",
                    "split",
                    "image",
                    "label_file",
                    "class_id",
                    "class_name",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "width",
                    "height",
                ],
            )
            if write_header:
                writer.writeheader()

            class_counts = {name: 0 for name in CLASSES}
            for index, entry in enumerate(entries):
                split = split_name(index, total)
                image_name = safe_name(entry.filename)
                image_path = OUT_DIR / "images" / split / image_name
                label_path = OUT_DIR / "labels" / split / (Path(image_name).stem + ".txt")

                if label_path.exists() and image_path.exists():
                    continue

                data, rgb = load_bytes_and_image(archive, entry)
                height, width = rgb.shape[:2]
                bbox = detect_meter_bbox(rgb)
                class_id = classify_meter(rgb, bbox)
                class_name = CLASSES[class_id]
                class_counts[class_name] += 1

                image_path.write_bytes(data)
                label_path.write_text(yolo_line(class_id, bbox, width, height), encoding="utf-8")

                x1, y1, x2, y2 = bbox
                writer.writerow(
                    {
                        "source": entry.filename,
                        "split": split,
                        "image": str(image_path),
                        "label_file": str(label_path),
                        "class_id": class_id,
                        "class_name": class_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": width,
                        "height": height,
                    }
                )

                if (index + 1) % 1000 == 0 or index + 1 == total:
                    print(f"processed {index + 1}/{total}")

    write_data_yaml()
    print("done")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
