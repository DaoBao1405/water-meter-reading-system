from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, Index, LargeBinary, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class MeterReading(Base):
    """One source image and its meter-reading result."""

    __tablename__ = "meter_readings"
    __table_args__ = (
        Index("ix_meter_readings_detected_at", "detected_at"),
        Index("ix_meter_readings_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    original_image_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    original_content_type: Mapped[str] = mapped_column(
        String(100), nullable=False, default="image/jpeg"
    )
    annotated_image_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    annotated_content_type: Mapped[str | None] = mapped_column(String(100))
    reading: Mapped[str | None] = mapped_column(String(100))
    corrected_reading: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    reading_direction: Mapped[str | None] = mapped_column(String(50))
    counter_box: Mapped[list[int] | None] = mapped_column(JSONB)
    counter_confidence: Mapped[float | None] = mapped_column(Float)
    average_digit_confidence: Mapped[float | None] = mapped_column(Float)
    image_width: Mapped[int] = mapped_column(nullable=False)
    image_height: Mapped[int] = mapped_column(nullable=False)
    liter_boxes: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    digits: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    ignored_liter_digits: Mapped[list[dict]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
