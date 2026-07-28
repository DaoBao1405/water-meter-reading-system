# Workflow đọc số đồng hồ nước bằng YOLO

Tài liệu này bổ sung một nhánh thị giác máy tính độc lập với pipeline LSTM cảnh báo áp lực nước hiện có.

## 1. Mục tiêu

Đầu vào là ảnh đồng hồ nước. Đầu ra là chuỗi số đọc được, ví dụ:

```json
{
  "reading": "0017.55",
  "main_value": "0017",
  "decimal_value": "55",
  "avg_digit_confidence": 0.91
}
```

## 2. Cấu trúc xử lý

```text
Ảnh đầu vào
  -> YOLO detect mặt đồng hồ / vùng số / từng chữ số
  -> lọc chữ số nằm trong digit_panel
  -> sắp xếp box từ trái sang phải
  -> ghép thành chuỗi số
  -> xuất JSON và ảnh kiểm tra
```

## 3. Tạo dataset YOLO

Tạo thư mục dataset:

```powershell
python scripts/meter_reader/make_dataset_dirs.py --dataset-dir data/meter_digits --scheme simple
```

Nếu đã tải dataset từ Roboflow ở dạng zip YOLOv8, import và chuẩn hóa đường dẫn bằng:

```powershell
python scripts/meter_reader/import_roboflow_dataset.py `
  --zip C:\Users\DELL\Downloads\DongHoFinal.v11i.yolov8.zip `
  --output-dir data/meter_digits_roboflow `
  --overwrite
```

Sau đó train bằng file:

```text
data/meter_digits_roboflow/data.yaml
```

Nếu muốn tách số đen và số đỏ ngay trong label:

```powershell
python scripts/meter_reader/make_dataset_dirs.py --dataset-dir data/meter_digits --scheme color
```

Schema `simple`:

```text
meter
digit_panel
digit_0 ... digit_9
```

Schema `color`:

```text
meter
digit_panel
black_0 ... black_9
red_0 ... red_9
```

Với ảnh như mẫu, nên label tối thiểu:

- `digit_panel`: vùng khung chứa các số.
- từng chữ số trong vùng hiển thị.
- `meter`: toàn bộ mặt đồng hồ, nếu bạn muốn model biết vị trí đồng hồ khi ảnh chụp rộng.

## 4. Gán nhãn

Dùng LabelImg, CVAT, Roboflow, hoặc makesense.ai và export định dạng YOLO.

Mỗi ảnh cần có file label cùng tên:

```text
data/meter_digits/images/train/img001.jpg
data/meter_digits/labels/train/img001.txt
```

Định dạng mỗi dòng YOLO:

```text
class_id x_center y_center width height
```

Các giá trị tọa độ đã được chuẩn hóa theo kích thước ảnh, từ `0` đến `1`.

## 5. Train YOLO

Cài dependencies:

```powershell
pip install -r requirements.txt
```

Train model:

```powershell
python scripts/meter_reader/train_yolo.py `
  --data data/meter_digits/data.yaml `
  --model yolo11n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 16
```

Nếu máy không có GPU:

```powershell
python scripts/meter_reader/train_yolo.py --data data/meter_digits/data.yaml --device cpu
```

Model tốt nhất thường nằm ở:

```text
runs/meter_reader/digits/weights/best.pt
```

## 6. Đọc số từ ảnh

Nếu dùng schema `color`, code tự tách `black_*` thành phần nguyên và `red_*` thành phần thập phân:

```powershell
python scripts/meter_reader/predict_reading.py `
  --weights runs/meter_reader/digits/weights/best.pt `
  --source path/to/meter.jpg `
  --save-vis outputs/meter_reader `
  --json-out outputs/meter_reader/results.json
```

Nếu dùng schema `simple` và muốn coi 2 số cuối là phần thập phân:

```powershell
python scripts/meter_reader/predict_reading.py `
  --weights runs/meter_reader/digits/weights/best.pt `
  --source path/to/meter.jpg `
  --decimal-digits 2 `
  --save-vis outputs/meter_reader `
  --json-out outputs/meter_reader/results.json
```

Nếu model chưa detect được `digit_panel`, có thể đọc toàn bộ chữ số trên ảnh:

```powershell
python scripts/meter_reader/predict_reading.py `
  --weights runs/meter_reader/digits/weights/best.pt `
  --source path/to/meter.jpg `
  --all-digits
```

## 7. Khuyến nghị dữ liệu

- Bắt đầu với 300-500 ảnh thật.
- Mỗi kiểu ánh sáng, góc nghiêng, độ bẩn, độ mờ nên có ảnh đại diện.
- Tách train/val/test theo tỷ lệ khoảng 80/10/10.
- Đánh giá bằng tỷ lệ đọc đúng cả chuỗi, không chỉ mAP.

## 8. Bước tích hợp tiếp theo

Sau khi model đọc ổn, có thể thêm bảng SQL lưu:

- đường dẫn ảnh
- chỉ số đọc được
- confidence
- thời gian đọc
- mã logger / mã đồng hồ

Từ đó pipeline cảnh báo áp lực nước hiện tại có thể dùng thêm dữ liệu chỉ số đồng hồ nước để đối chiếu lưu lượng hoặc phát hiện bất thường.
