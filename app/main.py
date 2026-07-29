from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from uuid import UUID

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Base, engine, get_db
from app.models import MeterReading
from app.pipeline import MeterReader, image_from_bytes
from app.schemas import (
    HealthResponse,
    MeterResponse,
    ReadingCorrection,
    ReadingDetail,
    ReadingSummary,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
COUNTER_WEIGHTS = Path(os.getenv("COUNTER_WEIGHTS", MODEL_DIR / "counter_best.pt"))
DIGIT_WEIGHTS = Path(os.getenv("DIGIT_WEIGHTS", MODEL_DIR / "digit_best.pt"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
EXPECTED_DIGIT_COUNT = int(os.getenv("EXPECTED_DIGIT_COUNT", "4"))


def configured_device() -> str | int:
    forced_device = os.getenv("MODEL_DEVICE")
    if forced_device:
        return int(forced_device) if forced_device.isdigit() else forced_device
    return 0 if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the database table and load YOLO models once at application startup."""
    Base.metadata.create_all(bind=engine)
    device = configured_device()
    app.state.reader = MeterReader.from_weights(
        counter_weights=COUNTER_WEIGHTS,
        digit_weights=DIGIT_WEIGHTS,
        device=device,
    )
    app.state.inference_lock = Lock()
    yield
    engine.dispose()


app = FastAPI(
    title="Water Meter Reader API",
    version="1.1.0",
    description="Runs two YOLO11 models and stores readings and image bytes in PostgreSQL.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


def get_reading_or_404(db: Session, reading_id: UUID) -> MeterReading:
    reading = db.get(MeterReading, reading_id)
    if reading is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    return reading


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    reader: MeterReader = request.app.state.reader
    return HealthResponse(status="ok", device=str(reader.device))


@app.post("/v1/meter/read", response_model=MeterResponse, tags=["inference"])
async def read_meter(
    request: Request,
    file: Annotated[UploadFile, File(description="Ảnh đồng hồ nước")],
    region_conf: Annotated[float, Query(ge=0.01, le=0.99)] = 0.25,
    digit_conf: Annotated[float, Query(ge=0.01, le=0.99)] = 0.25,
    rotate_degrees: Annotated[int, Query(ge=-180, le=180)] = 0,
    include_annotated_image: bool = False,
    db: Session = Depends(get_db),
) -> MeterResponse:
    """Run one image through the pipeline and store its images and result."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Chỉ nhận tệp ảnh.")

    raw_image = await file.read()
    if not raw_image:
        raise HTTPException(status_code=422, detail="Tệp ảnh trống.")
    if len(raw_image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 10 MB.")

    try:
        image = image_from_bytes(raw_image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reader: MeterReader = request.app.state.reader
    inference_lock: Lock = request.app.state.inference_lock

    def infer() -> dict[str, Any]:
        # GPU/YOLO instances should not be used by two requests at the same time.
        with inference_lock:
            return reader.predict(
                full_image=image,
                region_conf=region_conf,
                digit_conf=digit_conf,
                rotate_degrees=rotate_degrees,
                expected_digit_count=EXPECTED_DIGIT_COUNT,
            )

    result = await run_in_threadpool(infer)
    annotated_source = (
        image.rotate(rotate_degrees, expand=True) if rotate_degrees else image
    )
    annotated = await run_in_threadpool(reader.annotate, annotated_source, result)

    record = MeterReading(
        original_image_data=raw_image,
        original_content_type=file.content_type or "image/jpeg",
        annotated_image_data=annotated,
        annotated_content_type="image/png",
        reading=result.get("reading"),
        status=result["status"],
        reading_direction=result.get("reading_direction"),
        counter_box=result.get("counter_box"),
        counter_confidence=result.get("counter_confidence"),
        average_digit_confidence=result.get("average_digit_confidence"),
        image_width=result["image_size"]["width"],
        image_height=result["image_size"]["height"],
        liter_boxes=result.get("liter_boxes", []),
        digits=result.get("digits", []),
        ignored_liter_digits=result.get("ignored_liter_digits", []),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    response_data = {**result, "id": record.id, "detected_at": record.detected_at}
    if include_annotated_image:
        response_data["annotated_image_base64"] = base64.b64encode(annotated).decode(
            "ascii"
        )
    return MeterResponse(**response_data)


@app.get("/v1/readings", response_model=list[ReadingSummary], tags=["readings"])
def list_readings(
    db: Session = Depends(get_db),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[MeterReading]:
    statement = (
        select(MeterReading)
        .order_by(MeterReading.detected_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(statement))


@app.get("/v1/readings/{reading_id}", response_model=ReadingDetail, tags=["readings"])
def get_reading(reading_id: UUID, db: Session = Depends(get_db)) -> MeterReading:
    return get_reading_or_404(db, reading_id)


@app.get("/v1/readings/{reading_id}/image", tags=["readings"])
def get_original_image(reading_id: UUID, db: Session = Depends(get_db)) -> Response:
    reading = get_reading_or_404(db, reading_id)
    return Response(
        content=bytes(reading.original_image_data),
        media_type=reading.original_content_type,
    )


@app.get("/v1/readings/{reading_id}/annotated-image", tags=["readings"])
def get_annotated_image(reading_id: UUID, db: Session = Depends(get_db)) -> Response:
    reading = get_reading_or_404(db, reading_id)
    if reading.annotated_image_data is None:
        raise HTTPException(status_code=404, detail="Chưa có ảnh nhận diện.")
    return Response(
        content=bytes(reading.annotated_image_data),
        media_type=reading.annotated_content_type or "image/png",
    )


@app.patch(
    "/v1/readings/{reading_id}",
    response_model=ReadingDetail,
    tags=["readings"],
)
def correct_reading(
    reading_id: UUID,
    payload: ReadingCorrection,
    db: Session = Depends(get_db),
) -> MeterReading:
    reading = get_reading_or_404(db, reading_id)
    reading.corrected_reading = payload.corrected_reading
    db.commit()
    db.refresh(reading)
    return reading
