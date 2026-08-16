from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UpscaleModelType(StrEnum):
    ESRGAN = "esrgan"
    HAT = "hat"


class CropBox(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


class JobRecord(BaseModel):
    job_id: UUID
    status: JobStatus = JobStatus.QUEUED
    callback_url: AnyHttpUrl | None = None
    input_path: Path
    result_path: Path
    tile: int
    crop: CropBox | None = None
    model_type: UpscaleModelType = UpscaleModelType.ESRGAN
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    duration_seconds: float | None = None
    result_url: str | None = None
    error: str | None = None
    callback_delivered: bool = False
    callback_error: str | None = None


class JobAccepted(BaseModel):
    job_id: UUID
    status: JobStatus
    status_url: str


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    status_url: str
    result_url: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    duration_seconds: float | None = None
    error: str | None = None
    model_type: UpscaleModelType = UpscaleModelType.ESRGAN
    callback_delivered: bool
    callback_error: str | None = None


class WebhookPayload(BaseModel):
    job_id: UUID
    status: JobStatus
    result_url: str | None = None
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    duration_seconds: float | None = None
    completed_at: datetime
    error: str | None = None
    model_type: UpscaleModelType = UpscaleModelType.ESRGAN


class ErrorResponse(BaseModel):
    detail: str
