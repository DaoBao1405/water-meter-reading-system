# Train YOLO đọc đồng hồ nước trên Kaggle

## 1. Chuẩn bị Kaggle Notebook

Trong Kaggle:

1. Tạo Notebook mới.
2. Bật GPU: `Notebook options` -> `Accelerator` -> chọn `GPU T4 x2` hoặc `GPU P100`.
3. Upload file Roboflow zip thành Kaggle Dataset, ví dụ `DongHoFinal.v11i.yolov8.zip`.
4. Add Dataset đó vào Notebook.

Giả sử Kaggle mount dataset ở:

```text
/kaggle/input/donghofinal-yolov8/DongHoFinal.v11i.yolov8.zip
```

Nếu tên khác, sửa biến `ZIP_PATH` ở cell bên dưới.

## 2. Cài thư viện

```python
!pip install -q ultralytics pyyaml
```

## 3. Giải nén và sửa `data.yaml`

```python
from pathlib import Path
import zipfile
import yaml

ZIP_PATH = Path("/kaggle/input/donghofinal-yolov8/DongHoFinal.v11i.yolov8.zip")
DATASET_DIR = Path("/kaggle/working/meter_digits_roboflow")

DATASET_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH) as archive:
    archive.extractall(DATASET_DIR)

data_yaml = DATASET_DIR / "data.yaml"
data = yaml.safe_load(data_yaml.read_text())

normalized = {
    "path": str(DATASET_DIR),
    "train": "train/images",
    "val": "valid/images",
    "names": data["names"],
}

if (DATASET_DIR / "test" / "images").exists():
    normalized["test"] = "test/images"

if "nc" in data:
    normalized["nc"] = data["nc"]

data_yaml.write_text(yaml.safe_dump(normalized, sort_keys=False))
print(data_yaml.read_text())
```

## 4. Kiểm tra số lượng ảnh

```python
for split in ["train", "valid", "test"]:
    for kind in ["images", "labels"]:
        path = DATASET_DIR / split / kind
        if path.exists():
            print(f"{split}/{kind}", len(list(path.glob("*"))))
```

Kỳ vọng với file bạn đã tải:

```text
train/images 12725
train/labels 12725
valid/images 1997
valid/labels 1997
```

## 5. Train YOLO

Train nhanh để kiểm tra pipeline trước:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(DATASET_DIR / "data.yaml"),
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project="/kaggle/working/runs/meter_reader",
    name="digits_yolo11n",
    patience=10,
)
```

Train kỹ hơn:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data=str(DATASET_DIR / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="/kaggle/working/runs/meter_reader",
    name="digits_yolo11s",
    patience=20,
)
```

Nếu Kaggle báo hết VRAM, giảm:

```python
batch=8
```

hoặc dùng:

```python
model = YOLO("yolo11n.pt")
```

## 6. Xem model tốt nhất

```python
best_pt = Path("/kaggle/working/runs/meter_reader/digits_yolo11n/weights/best.pt")
print(best_pt, best_pt.exists())
```

Nếu bạn train bằng `digits_yolo11s`, đổi path thành:

```python
best_pt = Path("/kaggle/working/runs/meter_reader/digits_yolo11s/weights/best.pt")
```

## 7. Test predict trên vài ảnh validation

```python
from ultralytics import YOLO

model = YOLO(str(best_pt))
sample_images = list((DATASET_DIR / "valid" / "images").glob("*"))[:10]

preds = model.predict(
    source=[str(path) for path in sample_images],
    conf=0.25,
    imgsz=640,
    save=True,
    project="/kaggle/working/predictions",
    name="valid_samples",
)
```

Ảnh predict sẽ nằm trong:

```text
/kaggle/working/predictions/valid_samples
```

## 8. Nén output để download

```python
!zip -r /kaggle/working/meter_reader_training_outputs.zip /kaggle/working/runs/meter_reader /kaggle/working/predictions
```

Sau đó ở tab bên phải của Kaggle Notebook, phần `Output`, download:

```text
meter_reader_training_outputs.zip
```

File quan trọng nhất là:

```text
runs/meter_reader/.../weights/best.pt
```

## 9. Dùng model Kaggle ở máy local

Copy `best.pt` về project local, ví dụ:

```text
D:\Water_Pressure_Alerting\models\meter_reader\best.pt
```

Sau đó predict local:

```powershell
python scripts/meter_reader/predict_reading.py `
  --weights models/meter_reader/best.pt `
  --source path/to/meter.jpg `
  --decimal-digits 2 `
  --save-vis outputs/meter_reader `
  --json-out outputs/meter_reader/results.json
```
