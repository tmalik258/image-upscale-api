import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.routes.upscale import router as upscale_router
from app.services.job_service import Upscaler, UpscalingJobService
from app.services.upscaler import ImageUpscaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app(
    settings: Settings | None = None,
    upscaler: Upscaler | None = None,
    callback_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        app_settings.prepare_directories()
        active_upscaler = upscaler
        if active_upscaler is None:
            active_upscaler = await asyncio.to_thread(
                ImageUpscaler,
                app_settings.model_path,
                app_settings.model_blocks,
            )
        job_service = UpscalingJobService(
            settings=app_settings,
            upscaler=active_upscaler,
            callback_client=callback_client,
        )
        application.state.job_service = job_service
        job_service.start()
        yield
        await job_service.stop()

    application = FastAPI(
        title=app_settings.app_name,
        description=(
            "Submit images for 4x Real-ESRGAN upscaling and receive the result "
            "URL through an n8n webhook."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(upscale_router, prefix="/api/v1")
    application.add_exception_handler(HTTPException, _http_error)
    application.add_exception_handler(RequestValidationError, _validation_error)

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": app_settings.app_name}

    return application


async def _http_error(_: Request, error: Exception) -> JSONResponse:
    http_error = error
    if not isinstance(http_error, HTTPException):
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(
        status_code=http_error.status_code,
        content={"detail": str(http_error.detail)},
        headers=http_error.headers,
    )


async def _validation_error(_: Request, error: Exception) -> JSONResponse:
    validation_error = error
    if not isinstance(validation_error, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})
    messages = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in validation_error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


app = create_app()
