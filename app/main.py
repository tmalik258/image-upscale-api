import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.routes.dpi import router as dpi_router
from app.routes.upscale import router as upscale_router
from app.services.job_service import Upscaler, UpscalingJobService
from app.services.upscaler import ImageUpscaler


def create_app(
    settings: Settings | None = None,
    upscaler: Upscaler | None = None,
    callback_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        app_settings.prepare_directories()
        configure_logging(app_settings.log_dir)
        startup_logger = logging.getLogger("app.startup")
        startup_logger.info("Startup: directories and logging ready")
        active_upscaler = upscaler
        if active_upscaler is None:
            startup_logger.info("Startup: initializing upscaler (CUDA probe deferred)")
            active_upscaler = await asyncio.to_thread(
                ImageUpscaler,
                app_settings.weights_dir,
            )
            startup_logger.info("Startup: upscaler ready")
        job_service = UpscalingJobService(
            settings=app_settings,
            upscaler=active_upscaler,
            callback_client=callback_client,
        )
        application.state.settings = app_settings
        application.state.job_service = job_service
        job_service.start()
        startup_logger.info("Startup: job worker started")
        yield
        await job_service.stop()

    application = FastAPI(
        title=app_settings.app_name,
        description=(
            "Submit images for Real-ESRGAN or HAT upscaling and receive the result "
            "URL through an n8n webhook."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(upscale_router, prefix="/api/v1")
    application.include_router(dpi_router, prefix="/api/v1")
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
