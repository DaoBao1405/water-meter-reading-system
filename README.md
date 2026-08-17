# Hệ thống đọc chỉ số đồng hồ nước

Ứng dụng nhận diện chỉ số đồng hồ nước từ ảnh bằng hai model YOLO11, lưu kết quả vào PostgreSQL và cung cấp giao diện web để xem, kiểm tra và sửa chỉ số.

Phiên bản hiện tại sử dụng bộ model v2:

- Region model `region_itron_v2`: phát hiện vùng `counter` và `liter`.
- Digit model `digit_itron_aichi_v2`: phát hiện các chữ số từ `0` đến `9`.
- Pipeline có padding vùng counter, mask vùng liter và sắp xếp chữ số bằng PCA để hỗ trợ ảnh nằm ngang, dọc hoặc chéo.

## Tính năng

- Tải lên một hoặc nhiều ảnh từ giao diện web.
- Xoay ảnh trước khi nhận diện: `0°`, `90°`, `180°` hoặc `270°`.
- Điều chỉnh confidence của model chữ số.
- Nhận diện theo hai giai đoạn: vùng đồng hồ rồi đến chữ số.
- Trả ảnh PNG đã vẽ bounding box.
- Lưu ảnh gốc, ảnh chú thích và kết quả vào PostgreSQL.
- Xem 10 bản ghi gần nhất trên giao diện.
- Xác nhận hoặc sửa thủ công chỉ số đã nhận diện.
- Chạy bằng Docker Compose hoặc trực tiếp trong môi trường Python.

## Kiến trúc

```mermaid
flowchart LR
    U["Người dùng"] --> F["Frontend / Nginx<br/>localhost:3000"]
    F --> A["FastAPI<br/>localhost:8000"]
    A --> R["YOLO Region v2<br/>counter / liter"]
    R --> D["YOLO Digit v2<br/>0–9"]
    A --> P["PostgreSQL 16<br/>localhost:5432"]
    A --> F
```

Ba service trong `docker-compose.yml`:

| Service | Công nghệ | Cổng máy host | Vai trò |
| --- | --- | --- | --- |
| `frontend` | Nginx Alpine | `3000` | Phục vụ giao diện HTML/CSS/JavaScript |
| `api` | FastAPI + Uvicorn | `8000` | Nhận ảnh, chạy YOLO và quản lý kết quả |
| `postgres` | PostgreSQL 16 Alpine | `5432` | Lưu lịch sử, ảnh và kết quả nhận diện |

## Luồng nhận diện v2

1. Đọc ảnh, chuẩn hóa EXIF orientation và chuyển sang RGB.
2. Xoay ảnh nếu request có `rotate_degrees`.
3. Region model phát hiện `counter` và `liter` trên ảnh gốc với `imgsz=640`.
4. Chọn vùng `counter` có confidence cao nhất.
5. Mở rộng bounding box counter thêm 4% mỗi chiều rồi crop.
6. Đổi tọa độ các vùng `liter` sang hệ tọa độ của counter crop.
7. Che vùng liter bằng màu xám, mở rộng mask thêm 4 px.
8. Digit model phát hiện chữ số `0–9` trên counter crop.
9. Bỏ các chữ số có tâm nằm trong vùng liter.
10. Dùng PCA để sắp xếp chữ số dọc theo trục hiển thị và hướng về phía liter.
11. Chỉ trả `status="ok"` khi số chữ số đúng bằng `EXPECTED_DIGIT_COUNT`.

Giá trị `reading_direction` có dạng `pca_axis_X_Y`, ví dụ `pca_axis_1.000_0.000`. Nếu có không quá một chữ số, giá trị là `unknown`.

## Model bắt buộc

Đặt hai weights tại:

```text
models/
├── counter_best.pt
└── digit_best.pt
```

| File | Dataset v2 | Classes |
| --- | --- | --- |
| `models/counter_best.pt` | `region_itron_v2` | `counter`, `liter` |
| `models/digit_best.pt` | `digit_itron_aichi_v2` | `0` đến `9` |

Thư mục `models/*` được Git bỏ qua để tránh commit file nhị phân lớn. Chỉ `models/.gitkeep` được theo dõi. Vì Dockerfile copy weights vào image, sau khi thay model phải build lại API image.

## Cấu trúc dự án

```text
water-meter-reading-system/
├── app/
│   ├── main.py              # FastAPI routes và vòng đời ứng dụng
│   ├── pipeline.py          # Pipeline YOLO11 v2
│   ├── schemas.py           # Pydantic response/request schemas
│   ├── models.py            # SQLAlchemy model meter_readings
│   └── db.py                # PostgreSQL engine và session
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── nginx.conf
│   └── Dockerfile
├── models/
│   ├── counter_best.pt
│   └── digit_best.pt
├── tests/
│   └── test_pipeline.py
├── detection-yolo11.ipynb  # Chuẩn bị dữ liệu, train, đánh giá và export model
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Khởi chạy nhanh bằng Docker

### Yêu cầu

- Docker Desktop có Docker Compose.
- Hai model v2 đã được đặt đúng trong thư mục `models/`.
- Các cổng `3000`, `8000` và `5432` đang trống.

### 1. Tạo file môi trường

PowerShell:

```powershell
Copy-Item .env.example .env
```

Mở `.env` và đổi `POSTGRES_PASSWORD` trước khi sử dụng ngoài môi trường phát triển.

### 2. Kiểm tra weights

```powershell
Test-Path .\models\counter_best.pt
Test-Path .\models\digit_best.pt
```

Cả hai lệnh phải trả về `True`.

### 3. Build và chạy

```powershell
docker compose up --build -d
```

Sau khi các container khởi động:

- Giao diện: <http://localhost:3000>
- API health: <http://localhost:8000/health>
- Swagger UI: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

Kiểm tra trạng thái và log:

```powershell
docker compose ps
docker compose logs -f api
```

Dừng hệ thống nhưng giữ dữ liệu PostgreSQL:

```powershell
docker compose down
```

> `docker compose down -v` sẽ xóa volume PostgreSQL và toàn bộ lịch sử đã lưu.

## Chạy trực tiếp để phát triển

Các lệnh dưới đây dành cho PowerShell trên Windows.

### 1. Chạy PostgreSQL

Có thể chỉ chạy database bằng Docker:

```powershell
docker compose up -d postgres
```

### 2. Tạo môi trường Python

Yêu cầu Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dự án cố định `ultralytics==8.4.102` để môi trường inference khớp môi trường huấn luyện model v2.

### 3. Cấu hình và chạy API

```powershell
$env:DATABASE_URL = "postgresql+psycopg://meter:meter_dev_password@localhost:5432/meter"
$env:MODEL_DEVICE = "cpu"
$env:EXPECTED_DIGIT_COUNT = "4"
$env:CORS_ORIGINS = "http://localhost:3000"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Nếu muốn dùng GPU ngoài Docker và PyTorch nhận diện được CUDA:

```powershell
$env:MODEL_DEVICE = "0"
```

### 4. Chạy frontend

Mở terminal PowerShell khác:

```powershell
python -m http.server 3000 --directory frontend
```

Sau đó truy cập <http://localhost:3000>.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://meter:meter_dev_password@localhost:5432/meter` | Chuỗi kết nối PostgreSQL của API |
| `POSTGRES_PASSWORD` | `meter_dev_password` trong Compose | Mật khẩu PostgreSQL; nên đổi trong `.env` |
| `MODEL_DIR` | `models`; trong image là `/app/models` | Thư mục chứa weights |
| `COUNTER_WEIGHTS` | `${MODEL_DIR}/counter_best.pt` | Đường dẫn region model |
| `DIGIT_WEIGHTS` | `${MODEL_DIR}/digit_best.pt` | Đường dẫn digit model |
| `MODEL_DEVICE` | Tự chọn CUDA nếu có; Compose dùng `cpu` | Thiết bị inference: `cpu`, `0` hoặc `cuda:0` |
| `EXPECTED_DIGIT_COUNT` | `4` | Số chữ số chính xác để kết quả có trạng thái `ok` |
| `CORS_ORIGINS` | `*` trong code; Compose dùng `http://localhost:3000` | Danh sách origin được phép, phân cách bằng dấu phẩy |

Dockerfile hiện dùng image Python CPU và Compose chưa cấu hình GPU passthrough. Không đặt `MODEL_DEVICE=0` cho container hiện tại nếu chưa đổi image PyTorch và cấu hình GPU cho Docker.

## REST API

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Kiểm tra API và thiết bị inference |
| `POST` | `/v1/meter/read` | Nhận diện một ảnh và lưu kết quả |
| `GET` | `/v1/readings` | Lấy danh sách kết quả theo phân trang |
| `GET` | `/v1/readings/{reading_id}` | Lấy chi tiết một kết quả |
| `GET` | `/v1/readings/{reading_id}/image` | Trả ảnh gốc |
| `GET` | `/v1/readings/{reading_id}/annotated-image` | Trả ảnh PNG đã chú thích |
| `PATCH` | `/v1/readings/{reading_id}` | Xác nhận, sửa hoặc xóa chỉ số đã sửa |

### Health check

```powershell
curl.exe http://localhost:8000/health
```

Ví dụ:

```json
{
  "status": "ok",
  "device": "cpu"
}
```

### Nhận diện một ảnh

`POST /v1/meter/read` nhận `multipart/form-data` với trường `file`. API chỉ nhận tệp ảnh và giới hạn kích thước 10 MiB.

| Query parameter | Mặc định | Giới hạn | Ý nghĩa |
| --- | --- | --- | --- |
| `region_conf` | `0.25` | `0.01–0.99` | Confidence của region model |
| `digit_conf` | `0.25` | `0.01–0.99` | Confidence của digit model |
| `rotate_degrees` | `0` | `-180–180` | Góc xoay trước khi inference |
| `include_annotated_image` | `false` | Boolean | Thêm ảnh PNG dạng Base64 vào response |

Giao diện web đặt `digit_conf=0.60` theo mặc định và gửi từng ảnh tuần tự. API xử lý một ảnh cho mỗi request.

PowerShell:

```powershell
curl.exe -X POST `
  "http://localhost:8000/v1/meter/read?region_conf=0.25&digit_conf=0.60&rotate_degrees=0" `
  -F "file=@C:\duong-dan\anh-dong-ho.jpg"
```

Ví dụ response rút gọn:

```json
{
  "id": "9b265fc9-37b7-43e9-830d-0bd584c35112",
  "status": "ok",
  "reading": "1234",
  "reading_direction": "pca_axis_1.000_0.000",
  "counter_box": [115, 74, 515, 256],
  "counter_confidence": 0.963,
  "average_digit_confidence": 0.912,
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "detected_at": "2026-08-12T06:30:00Z",
  "liter_boxes": [],
  "liter_boxes_in_crop": [],
  "digits": [],
  "ignored_liter_digits": [],
  "annotated_image_base64": null
}
```

`digits` trong response thực tế chứa từng chữ số, bounding box cục bộ, tâm box và confidence. Mảng được rút gọn trong ví dụ trên.

### Trạng thái nhận diện

| Status | Ý nghĩa | Giá trị `reading` |
| --- | --- | --- |
| `ok` | Phát hiện đúng `EXPECTED_DIGIT_COUNT` chữ số | Chuỗi chỉ số |
| `counter_not_found` | Không tìm thấy vùng counter | `null` |
| `invalid_counter_box` | Bounding box counter không hợp lệ | `null` |
| `digits_not_found` | Có counter nhưng không có chữ số được chấp nhận | `null` |
| `unexpected_digit_count` | Số chữ số khác `EXPECTED_DIGIT_COUNT` | `null` |

Các bounding box chữ số vẫn được giữ trong response khi số lượng không đúng để phục vụ kiểm tra thủ công.

### Lịch sử và phân trang

```http
GET /v1/readings?limit=20&offset=0
```

- `limit`: từ `1` đến `100`, mặc định `20`.
- `offset`: từ `0`, mặc định `0`.
- Bản ghi được sắp xếp mới nhất trước.
- Danh sách không chứa bytes ảnh; dùng endpoint ảnh riêng khi cần.

### Xác nhận hoặc sửa chỉ số

```powershell
curl.exe -X PATCH `
  "http://localhost:8000/v1/readings/READING_ID" `
  -H "Content-Type: application/json" `
  -d '{"corrected_reading":"1234"}'
```

Xóa giá trị đã sửa:

```powershell
curl.exe -X PATCH `
  "http://localhost:8000/v1/readings/READING_ID" `
  -H "Content-Type: application/json" `
  -d '{"corrected_reading":null}'
```

`corrected_reading` có độ dài tối đa 100 ký tự. Backend hiện không bắt buộc giá trị này chỉ chứa chữ số.

## Dữ liệu PostgreSQL

Bảng `meter_readings` lưu:

- UUID và thời điểm nhận diện.
- Ảnh gốc cùng content type.
- Ảnh PNG đã chú thích cùng content type.
- Chỉ số nhận diện và chỉ số đã sửa.
- Trạng thái, hướng đọc và confidence.
- Kích thước ảnh và bounding box counter.
- Danh sách vùng liter, chữ số được chấp nhận và chữ số bị loại trong liter.

Schema được tạo tự động khi API khởi động. Dữ liệu Docker được lưu trong volume `postgres_data`.

## Kiểm thử

Chạy toàn bộ kiểm thử pipeline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra biên dịch Python:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
```

Bộ test hiện kiểm tra:

- Counter crop được mở rộng đúng 4%.
- Chỉ số chỉ hợp lệ khi có đúng số chữ số yêu cầu.
- PCA sắp xếp đúng dãy chữ số nằm chéo theo hướng liter.

## Huấn luyện và thay model

Notebook `detection-yolo11.ipynb` bao gồm quy trình:

1. Chuẩn bị dataset Itron v2 và Aichi.
2. Chuẩn hóa mapping class.
3. Tạo dataset region `counter/liter` và dataset digit `0–9`.
4. Train `region_itron_v2` từ YOLO11n.
5. Train `digit_itron_aichi_v2` từ YOLO11s.
6. Đánh giá Precision, Recall, mAP và ảnh validation.
7. Export hai weights thành `counter_best.pt` và `digit_best.pt`.

Sau khi tải weights mới:

```powershell
Copy-Item C:\duong-dan\counter_best.pt .\models\counter_best.pt -Force
Copy-Item C:\duong-dan\digit_best.pt .\models\digit_best.pt -Force
```

Nếu chạy Docker:

```powershell
docker compose up --build -d
```

Nếu chạy Uvicorn trực tiếp, dừng và khởi động lại API vì model chỉ được load một lần trong vòng đời ứng dụng.

## Xử lý lỗi thường gặp

### API báo không tìm thấy weights

Kiểm tra:

```powershell
Get-Item .\models\counter_best.pt
Get-Item .\models\digit_best.pt
```

Tên file phải chính xác. Nếu vừa thay model trong Docker, cần build lại image.

### Frontend không tải được lịch sử

- Kiểm tra <http://localhost:8000/health>.
- Kiểm tra `CORS_ORIGINS` có chứa `http://localhost:3000`.
- Xem log bằng `docker compose logs -f api`.
- Frontend mặc định gọi API tại `http://localhost:8000`.

### API không kết nối được PostgreSQL

```powershell
docker compose ps postgres
docker compose logs postgres
```

Khi chạy API ngoài Docker, hostname trong `DATABASE_URL` phải là `localhost`. Khi chạy trong Compose, hostname phải là `postgres`.

### Kết quả là `unexpected_digit_count`

- Thử giảm `digit_conf` nếu thiếu chữ số.
- Thử tăng `digit_conf` nếu có nhiều box nhiễu.
- Chọn đúng góc xoay ảnh.
- Kiểm tra vùng counter và liter trên ảnh chú thích.
- Xác nhận `EXPECTED_DIGIT_COUNT` phù hợp loại đồng hồ.

### Kết quả sai thứ tự

- Kiểm tra region model có phát hiện đúng vùng liter không.
- Kiểm tra ảnh bị xoay hoặc phản chiếu bất thường.
- Xem `reading_direction` và vị trí các bounding box chữ số trong response.

## Lưu ý triển khai production

- Đổi mật khẩu PostgreSQL và không commit file `.env`.
- Giới hạn `CORS_ORIGINS` theo domain thật.
- Đặt reverse proxy HTTPS trước frontend và API.
- Không công khai cổng PostgreSQL ra Internet.
- Sao lưu volume PostgreSQL định kỳ.
- Lưu weights trong artifact storage hoặc image registry phù hợp; không commit trực tiếp vào Git.
- Pipeline khóa inference để tránh hai request dùng cùng YOLO/GPU đồng thời; cần thiết kế worker/GPU riêng nếu muốn tăng throughput.
- API hiện lưu toàn bộ bytes ảnh trong PostgreSQL; cần theo dõi dung lượng khi triển khai lâu dài.

## Công nghệ

- Python 3.11
- FastAPI và Uvicorn
- Ultralytics YOLO11 `8.4.102`
- PyTorch
- Pillow và NumPy
- SQLAlchemy 2 và Psycopg 3
- PostgreSQL 16
- HTML, CSS và JavaScript thuần
- Nginx
- Docker Compose
