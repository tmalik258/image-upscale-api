import asyncio
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError
from pydantic import AnyHttpUrl

from app.api_schema.jobs import (
    ErrorResponse,
    JobAccepted,
    JobResponse,
    JobStatus,
)
from app.services.job_service import UpscalingJobService
from app.utils.crops import build_crop
from app.utils.uploads import (
    ALLOWED_IMAGE_TYPES,
    download_image,
    save_upload,
    verify_image,
)


router = APIRouter(prefix="/upscale", tags=["upscaling"])


def get_job_service(request: Request) -> UpscalingJobService:
    return request.app.state.job_service


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def create_upscaling_job(
    request: Request,
    callback_url: Annotated[AnyHttpUrl | None, Form()] = None,
    image: Annotated[
        UploadFile | None,
        File(description="JPEG, PNG, or WebP image"),
    ] = None,
    image_url: Annotated[AnyHttpUrl | None, Form()] = None,
    tile: Annotated[int | None, Form(ge=0, le=4096)] = None,
    crop_left: Annotated[int | None, Form(ge=0)] = None,
    crop_top: Annotated[int | None, Form(ge=0)] = None,
    crop_right: Annotated[int | None, Form(gt=0)] = None,
    crop_bottom: Annotated[int | None, Form(gt=0)] = None,
) -> JobAccepted:
    service = get_job_service(request)
    settings = service.settings
    if (image is None) == (image_url is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide exactly one of image or image_url",
        )
    if image is not None and image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be JPEG, PNG, or WebP",
        )

    crop = build_crop(crop_left, crop_top, crop_right, crop_bottom)
    input_path = settings.upload_dir / f"{uuid4()}.upload"
    try:
        if image is not None:
            await save_upload(image, input_path, settings.max_upload_bytes)
        else:
            await download_image(
                str(image_url),
                input_path,
                settings.max_upload_bytes,
                settings.image_download_timeout_seconds,
            )
        await asyncio.to_thread(verify_image, input_path)
    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError) as error:
        input_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image source is not a valid image",
        ) from error
    finally:
        if image is not None:
            await image.close()

    return await service.submit(
        input_path=input_path,
        callback_url=str(callback_url) if callback_url is not None else None,
        tile=settings.default_tile if tile is None else tile,
        crop=crop,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_upscaling_job(request: Request, job_id: UUID) -> JobResponse:
    service = get_job_service(request)
    record = service.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upscaling job not found",
        )
    return service.response_for(record)


@router.get(
    "/results/{job_id}.png",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def download_upscaled_image(request: Request, job_id: UUID) -> FileResponse:
    service = get_job_service(request)
    record = service.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upscaling job not found",
        )
    if record.status is not JobStatus.COMPLETED or not record.result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Upscaled image is not ready",
        )
    return FileResponse(
        record.result_path,
        media_type="image/png",
        filename=f"upscaled-{job_id}.png",
    )
