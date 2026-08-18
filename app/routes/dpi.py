import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from app.api_schema.dpi import DpiRewriteRequest, DpiRewriteResponse
from app.api_schema.jobs import ErrorResponse
from app.config import Settings
from app.services.dpi import rewrite_image_dpi
from app.utils.dpi import MEDIA_TYPES, DpiOutputFormat
from app.utils.uploads import download_image, verify_image


router = APIRouter(prefix="/dpi", tags=["dpi"])


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.post(
    "",
    response_model=DpiRewriteResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def rewrite_dpi(
    request: Request,
    body: DpiRewriteRequest,
) -> DpiRewriteResponse:
    settings = get_settings(request)
    image_id = uuid4()
    input_path = settings.upload_dir / f"{image_id}.upload"
    output_path = settings.result_dir / f"{image_id}.{body.format.value}"
    try:
        await download_image(
            str(body.image_url),
            input_path,
            settings.max_upload_bytes,
            settings.image_download_timeout_seconds,
        )
        await asyncio.to_thread(verify_image, input_path)
        result = await asyncio.to_thread(
            rewrite_image_dpi,
            input_path,
            output_path,
            body.dpi,
            body.format,
        )
    except HTTPException:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError) as error:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image source is not a valid image",
        ) from error
    input_path.unlink(missing_ok=True)
    return DpiRewriteResponse(
        result_url=settings.absolute_url(
            f"/api/v1/dpi/results/{image_id}.{body.format.value}"
        ),
        dpi=result.dpi,
        format=result.output_format,
        width=result.width,
        height=result.height,
    )


@router.get(
    "/results/{image_id}.jpg",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}},
)
async def download_dpi_jpeg(request: Request, image_id: UUID) -> FileResponse:
    return _dpi_file_response(request, image_id, DpiOutputFormat.JPG)


@router.get(
    "/results/{image_id}.png",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}},
)
async def download_dpi_png(request: Request, image_id: UUID) -> FileResponse:
    return _dpi_file_response(request, image_id, DpiOutputFormat.PNG)


def _dpi_file_response(
    request: Request,
    image_id: UUID,
    output_format: DpiOutputFormat,
) -> FileResponse:
    settings = get_settings(request)
    result_path = settings.result_dir / f"{image_id}.{output_format.value}"
    if not result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="DPI-rewritten image not found",
        )
    return FileResponse(
        result_path,
        media_type=MEDIA_TYPES[output_format],
        filename=f"dpi-{image_id}.{output_format.value}",
    )
