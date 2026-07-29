from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    device: str


class MeterResponse(BaseModel):
    id: UUID
    status: str
    reading: str | None = None
    reading_direction: str | None = None
    counter_box: list[int] | None = None
    counter_confidence: float | None = None
    average_digit_confidence: float | None = None
    image_size: dict[str, int]
    detected_at: datetime
    liter_boxes: list[dict[str, Any]] = Field(default_factory=list)
    liter_boxes_in_crop: list[dict[str, Any]] = Field(default_factory=list)
    digits: list[dict[str, Any]] = Field(default_factory=list)
    ignored_liter_digits: list[dict[str, Any]] = Field(default_factory=list)
    annotated_image_base64: str | None = None


class ReadingSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    reading: str | None
    corrected_reading: str | None
    status: str
    average_digit_confidence: float | None
    detected_at: datetime
    image_width: int
    image_height: int


class ReadingDetail(ReadingSummary):
    reading_direction: str | None
    counter_box: list[int] | None
    counter_confidence: float | None
    liter_boxes: list[dict[str, Any]]
    digits: list[dict[str, Any]]
    ignored_liter_digits: list[dict[str, Any]]


class ReadingCorrection(BaseModel):
    corrected_reading: str | None = Field(default=None, max_length=100)
