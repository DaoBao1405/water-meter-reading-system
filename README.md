# Hệ thống đọc chỉ số đồng hồ nước

Hệ thống nhận diện chỉ số đồng hồ nước từ ảnh bằng pipeline hai tầng YOLO11. Người dùng có thể tải một hoặc nhiều ảnh trên giao diện web, xem vùng nhận diện và chữ số, sửa kết quả khi cần, đồng thời tra cứu lịch sử đã lưu trong PostgreSQL.

## Tính năng chính

- Tải ảnh bằng hộp chọn tệp hoặc kéo thả; hỗ trợ xử lý nhiều ảnh lần lượt.
- Chuẩn hóa hướng ảnh theo EXIF và cho phép xoay ảnh trước khi nhận diện.
- Dùng model thứ nhất để phát hiện vùng hiển thị `counter` và vùng `liter`.
- Dùng model thứ hai để nhận diện các chữ số từ `0` đến `9`.
- Loại các chữ số thuộc vùng `liter` khỏi chỉ số chính.
- Tự xác định hướng đọc: trái sang phải, phải sang trái, trên xuống dưới hoặc dưới lên trên.
- Chỉ trả về chỉ số khi phát hiện đủ số chữ số tối thiểu được cấu hình.
- Sinh ảnh PNG có bounding box cho vùng đồng hồ, vùng lít và các chữ số được chấp nhận.
- Lưu ảnh gốc, ảnh chú thích, kết quả nhận diện và thời điểm xử lý trong PostgreSQL.
- Cho phép người dùng xác nhận hoặc sửa chỉ số đã nhận diện.
- Cung cấp REST API, Swagger UI và giao diện web responsive.

## Kiến trúc

```mermaid
flowchart LR
    U["Trình duyệt"] --> F["Frontend tĩnh<br/>Nginx :3000"]
    F --> A["FastAPI :8000"]
    A --> R["YOLO counter/liter"]
    R --> D["YOLO digit reader"]
    D --> A
    A --> P[("PostgreSQL :5432")]
```

Docker Compose khởi chạy ba dịch vụ:

| Dịch vụ | Công nghệ | Cổng | Vai trò |
| --- | --- | --- | --- |
| `frontend` | Nginx 1.27 Alpine | `3000` | Phục vụ HTML, CSS và JavaScript |
| `api` | FastAPI, Uvicorn, Ultralytics | `8000` | Nhận ảnh, chạy model và quản lý kết quả |
| `postgres` | PostgreSQL 16 Alpine | `5432` | Lưu metadata và dữ liệu nhị phân của ảnh |

## Luồng nhận diện

1. API kiểm tra loại tệp, kích thước tối đa 10 MB và tính hợp lệ của ảnh.
2. Pillow chuẩn hóa EXIF, chuyển ảnh sang RGB và xoay ảnh nếu có yêu cầu.
3. `counter_best.pt` phát hiện các vùng `counter` và `liter` trên ảnh đầy đủ.
4. Hệ thống chọn vùng `counter` có confidence cao nhất rồi crop vùng đó.
5. Các vùng `liter` giao với crop được chuyển về tọa độ cục bộ và che bằng màu xám.
6. `digit_best.pt` nhận diện chữ số trên crop đã che vùng lít.
7. Những chữ số có tâm nằm trong vùng `liter` tiếp tục bị loại để tránh tính phần thập phân/lít vào chỉ số chính.
8. Các chữ số còn lại được sắp xếp theo hướng tiến về vùng `liter`.
9. Hệ thống tạo ảnh chú thích, lưu kết quả cùng ảnh vào PostgreSQL và trả JSON cho client.

YOLO inference được bảo vệ bằng một lock trong tiến trình API, vì vậy hai request không dùng chung model/GPU đồng thời.

## Cấu trúc dự án

```text
water-meter-reading-system/
├── app/
│   ├── main.py          # FastAPI, routes, validation và lưu kết quả
│   ├── pipeline.py      # Pipeline YOLO hai tầng và vẽ ảnh chú thích
│   ├── db.py            # Kết nối và session SQLAlchemy
│   ├── models.py        # Model bảng meter_readings
│   └── schemas.py       # Pydantic request/response schemas
├── frontend/
│   ├── index.html       # Giao diện tải ảnh và lịch sử
│   ├── app.js           # Gọi API, render kết quả và sửa chỉ số
│   ├── styles.css       # Giao diện responsive
│   ├── nginx.conf       # Cấu hình phục vụ frontend
│   └── Dockerfile
├── models/
│   ├── counter_best.pt  # Model phát hiện counter/liter
│   └── digit_best.pt    # Model nhận diện chữ số 0–9
├── detection-yolo11.ipynb # Chuẩn bị dữ liệu, train, đánh giá và export model
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Yêu cầu

### Chạy bằng Docker

- Docker Desktop hoặc Docker Engine có Docker Compose.
- Hai file model phải tồn tại:

```text
models/counter_best.pt
models/digit_best.pt
```

Hai weights đã có sẵn trong phiên bản dự án hiện tại.

### Chạy trực tiếp

- Python 3.11 được khuyến nghị.
- PostgreSQL đang hoạt động.
- Các thư viện hệ thống cần thiết cho Pillow/OpenCV tùy theo hệ điều hành.

## Khởi chạy nhanh bằng Docker

### 1. Tạo cấu hình môi trường

PowerShell:

```powershell
Copy-Item .env.example .env
```

Đổi `POSTGRES_PASSWORD` trong `.env` trước khi dùng ngoài môi trường phát triển.

### 2. Build và chạy

```powershell
docker compose up --build
```

Sau khi các container sẵn sàng:

- Giao diện web: <http://localhost:3000>
- Swagger UI: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
- Health check: <http://localhost:8000/health>

Bảng `meter_readings` và các index được SQLAlchemy tự tạo khi API khởi động. Dữ liệu PostgreSQL được giữ trong Docker volume `postgres_data`.

Chạy nền:

```powershell
docker compose up --build -d
docker compose logs -f api
```

Dừng hệ thống nhưng giữ dữ liệu:

```powershell
docker compose down
```

> Không thêm `-v` vào lệnh `docker compose down` nếu muốn giữ lịch sử, vì tùy chọn đó sẽ xóa volume PostgreSQL.

## Chạy trực tiếp để phát triển

### 1. Khởi chạy PostgreSQL

Có thể chỉ chạy database bằng Docker:

```powershell
docker compose up -d postgres
```

### 2. Tạo môi trường Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cấu hình và chạy API

```powershell
$env:DATABASE_URL = "postgresql+psycopg://meter:meter_dev_password@localhost:5432/meter"
$env:MODEL_DEVICE = "cpu"
$env:EXPECTED_DIGIT_COUNT = "4"
$env:CORS_ORIGINS = "http://localhost:3000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Nếu password trong `.env` đã được đổi, phần password của `DATABASE_URL` cũng phải giống giá trị đó.

### 4. Chạy frontend

Mở terminal khác:

```powershell
python -m http.server 3000 --directory frontend
```

Sau đó truy cập <http://localhost:3000>.

Frontend mặc định gọi API tại `http://localhost:8000`. Có thể đặt `window.METER_API_BASE` trước khi tải `frontend/app.js` nếu triển khai API ở địa chỉ khác.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://meter:meter_dev_password@localhost:5432/meter` | Chuỗi kết nối PostgreSQL của API |
| `POSTGRES_PASSWORD` | `meter_dev_password` trong Compose | Password cho container PostgreSQL và chuỗi kết nối của API |
| `MODEL_DIR` | `models` (`/app/models` trong image) | Thư mục chứa weights |
| `COUNTER_WEIGHTS` | `${MODEL_DIR}/counter_best.pt` | Đường dẫn model phát hiện `counter`/`liter` |
| `DIGIT_WEIGHTS` | `${MODEL_DIR}/digit_best.pt` | Đường dẫn model chữ số |
| `MODEL_DEVICE` | Tự chọn CUDA nếu có; Compose dùng `cpu` | Thiết bị inference, ví dụ `cpu`, `0` hoặc `cuda:0` |
| `EXPECTED_DIGIT_COUNT` | `4` | Số chữ số tối thiểu để kết quả có trạng thái `ok` |
| `CORS_ORIGINS` | `*` trong code; Compose dùng `http://localhost:3000` | Danh sách origin được phép, phân cách bằng dấu phẩy |

Dockerfile hiện dùng image Python CPU và Compose chưa cấu hình GPU passthrough. Không đổi `MODEL_DEVICE` sang CUDA trong Docker nếu chưa thay image PyTorch và cấu hình GPU cho container.

## REST API

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Kiểm tra API và thiết bị inference |
| `POST` | `/v1/meter/read` | Nhận diện một ảnh và lưu kết quả |
| `GET` | `/v1/readings` | Lấy danh sách kết quả, không chứa bytes ảnh |
| `GET` | `/v1/readings/{reading_id}` | Lấy chi tiết một kết quả |
| `GET` | `/v1/readings/{reading_id}/image` | Trả ảnh gốc |
| `GET` | `/v1/readings/{reading_id}/annotated-image` | Trả ảnh PNG đã vẽ bounding box |
| `PATCH` | `/v1/readings/{reading_id}` | Gán hoặc xóa `corrected_reading` |

### `POST /v1/meter/read`

Request dùng `multipart/form-data` với trường `file`.

| Query parameter | Mặc định | Giới hạn | Ý nghĩa |
| --- | --- | --- | --- |
| `region_conf` | `0.25` | `0.01–0.99` | Confidence của model vùng |
| `digit_conf` | `0.25` | `0.01–0.99` | Confidence của model chữ số |
| `rotate_degrees` | `0` | `-180–180` | Góc xoay ảnh trước inference |
| `include_annotated_image` | `false` | Boolean | Trả thêm PNG dạng Base64 trong JSON |

Giao diện web đang đặt `digit_conf=0.60` và xử lý từng ảnh tuần tự. Giao diện có tùy chọn xoay `270°`, nhưng API hiện chỉ chấp nhận góc từ `-180°` đến `180°`; vì vậy tùy chọn `270°` sẽ bị API từ chối. Khi gọi API trực tiếp, dùng `-90°` thay cho `270°`.

Ví dụ PowerShell:

```powershell
curl.exe -X POST `
  "http://localhost:8000/v1/meter/read?region_conf=0.25&digit_conf=0.60&rotate_degrees=0" `
  -F "file=@C:\duong-dan\anh-dong-ho.jpg"
```

Ví dụ response thành công:

```json
{
  "id": "9b265fc9-37b7-43e9-830d-0bd584c35112",
  "status": "ok",
  "reading": "1234",
  "reading_direction": "left_to_right",
  "counter_box": [120, 80, 510, 250],
  "counter_confidence": 0.963,
  "average_digit_confidence": 0.912,
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "detected_at": "2026-07-29T10:00:00Z",
  "liter_boxes": [],
  "liter_boxes_in_crop": [],
  "digits": [],
  "ignored_liter_digits": [],
  "annotated_image_base64": null
}
```

Trường `digits` trong response thực tế chứa từng chữ số, bounding box cục bộ, tâm box và confidence. Mảng được rút gọn trong ví dụ để dễ đọc.

### Phân trang lịch sử

```http
GET /v1/readings?limit=20&offset=0
```

- `limit`: từ `1` đến `100`, mặc định `20`.
- `offset`: từ `0`, mặc định `0`.
- Kết quả được sắp xếp mới nhất trước.
- Frontend hiển thị 10 bản ghi gần nhất.

### Xác nhận hoặc sửa chỉ số

```powershell
curl.exe -X PATCH `
  "http://localhost:8000/v1/readings/READING_ID" `
  -H "Content-Type: application/json" `
  -d '{"corrected_reading":"1234"}'
```

Để xóa giá trị đã sửa:

```json
{
  "corrected_reading": null
}
```

`corrected_reading` có độ dài tối đa 100 ký tự. Code hiện tại không bắt buộc trường này chỉ chứa chữ số.

## Trạng thái nhận diện

| Status | Ý nghĩa |
| --- | --- |
| `ok` | Phát hiện ít nhất `EXPECTED_DIGIT_COUNT` chữ số và trả chỉ số trong `reading` |
| `counter_not_found` | Không tìm thấy vùng hiển thị đồng hồ |
| `invalid_counter_box` | Bounding box của vùng đồng hồ không hợp lệ |
| `digits_not_found` | Tìm thấy vùng đồng hồ nhưng không có chữ số được chấp nhận |
| `incomplete_digits` | Có chữ số nhưng ít hơn `EXPECTED_DIGIT_COUNT`; `reading` được trả về là `null` |

Hướng đọc có thể là:

- `left_to_right`
- `right_to_left`
- `top_to_bottom`
- `bottom_to_top`
- `unknown` khi có không quá một chữ số

## Dữ liệu PostgreSQL

Bảng `meter_readings` lưu:

- UUID của bản ghi.
- Ảnh gốc và content type.
- Ảnh PNG đã chú thích và content type.
- Chỉ số model đọc được và chỉ số người dùng sửa.
- Trạng thái và hướng đọc.
- Bounding box/confidence của vùng đồng hồ.
- Confidence chữ số trung bình.
- Kích thước ảnh.
- Các vùng `liter`, chữ số được chấp nhận và chữ số bị loại.
- Thông báo lỗi dự phòng và thời điểm nhận diện.

Hai index được tạo trên `detected_at` và `status`. Dự án lưu toàn bộ bytes ảnh trong PostgreSQL, không cần S3, MinIO hoặc MongoDB.

## Huấn luyện và export model

Notebook `detection-yolo11.ipynb` gồm 21 code cell và thực hiện toàn bộ quy trình:

1. Đọc và giải nén dataset YOLO từ `/mnt/yolo-data`.
2. Audit số lượng ảnh, class và phân bố bounding box cho các split `train`, `valid`, `test`.
3. Chuyển dataset nguồn thành hai dataset:
   - `counter_liter_detector`: hai class `counter` và `liter`.
   - `digit_reader_no_liter`: mười class chữ số `0–9`, crop theo vùng counter và che vùng liter.
4. Kiểm tra trực quan crop và nhãn trước khi train.
5. Train `yolo11n.pt` cho model vùng trong tối đa 120 epoch.
6. Train `yolo11s.pt` cho model chữ số trong tối đa 120 epoch.
7. So sánh loss, precision, recall và mAP; tạo biểu đồ validation.
8. Chạy thử pipeline trên ảnh ngoài dataset.
9. Export hai weights thành:

```text
counter_best.pt
digit_best.pt
```

Notebook được viết cho môi trường có volume `/mnt/yolo-data` và sử dụng IPython magic `%uv`; cần điều chỉnh đường dẫn nếu chạy trên máy cá nhân hoặc nền tảng notebook khác.

## Xử lý lỗi thường gặp

### API không khởi động vì thiếu weights

Kiểm tra:

```powershell
Test-Path .\models\counter_best.pt
Test-Path .\models\digit_best.pt
```

API load cả hai model ngay khi khởi động và sẽ dừng nếu thiếu một file.

### Frontend báo không thể tải lịch sử

- Kiểm tra <http://localhost:8000/health>.
- Kiểm tra API có chạy đúng cổng `8000`.
- Kiểm tra `CORS_ORIGINS` chứa origin của frontend.
- Nếu API không ở localhost, cấu hình lại `window.METER_API_BASE`.

### API không kết nối được PostgreSQL

```powershell
docker compose ps
docker compose logs postgres
docker compose logs api
```

Kiểm tra password trong `.env` và `DATABASE_URL` phải khớp nhau.

### Nhận diện thiếu chữ số

- Giảm `digit_conf` từng bước nhỏ.
- Thử xoay ảnh đúng chiều mặt đồng hồ.
- Dùng ảnh rõ nét, đủ sáng và chụp gần vùng hiển thị.
- Kiểm tra `EXPECTED_DIGIT_COUNT` có đúng loại đồng hồ hay không.

## Lưu ý khi triển khai production

Phiên bản hiện tại phù hợp cho demo hoặc mạng nội bộ. Trước khi công khai cần:

- Thêm xác thực và phân quyền cho API.
- Không dùng password mặc định.
- Giới hạn CORS theo domain thực tế.
- Bật HTTPS và đặt API sau reverse proxy.
- Thiết lập backup/retention vì ảnh nhị phân làm database tăng nhanh.
- Dùng migration tool như Alembic thay cho `create_all`.
- Bổ sung test tự động, logging, monitoring và rate limiting.
- Cân nhắc hàng đợi inference hoặc nhiều worker chuyên dụng khi cần xử lý đồng thời.

## Công nghệ sử dụng

- Python 3.11
- FastAPI và Uvicorn
- Ultralytics YOLO11, PyTorch
- Pillow và NumPy
- SQLAlchemy 2 và psycopg 3
- PostgreSQL 16
- HTML, CSS, JavaScript thuần
- Nginx
- Docker Compose
