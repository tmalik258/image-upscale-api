import json
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api_schema.jobs import UpscaleModelType
from app.config import Settings
from app.main import create_app
from app.services.upscaler import UpscaleResult


class FakeUpscaler:
    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
        model_type: UpscaleModelType = UpscaleModelType.ESRGAN,
        job_id: UUID | None = None,
    ) -> UpscaleResult:
        with Image.open(input_path) as source:
            image = source.convert("RGB")
        if crop:
            image = image.crop(crop)
        input_width, input_height = image.size
        output = image.resize((input_width * 4, input_height * 4))
        output.save(output_path, format="PNG")
        return UpscaleResult(
            input_width=input_width,
            input_height=input_height,
            output_width=output.width,
            output_height=output.height,
        )


class FailingUpscaler:
    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
        model_type: UpscaleModelType = UpscaleModelType.ESRGAN,
        job_id: UUID | None = None,
    ) -> UpscaleResult:
        raise RuntimeError("Test inference failure")


class GpuOomUpscaler:
    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
        model_type: UpscaleModelType = UpscaleModelType.ESRGAN,
        job_id: UUID | None = None,
    ) -> UpscaleResult:
        raise RuntimeError(
            f"GPU out of memory for {model_type.value} 2048x2048 tile={tile}: "
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )


@pytest.fixture
def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def api_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, list[dict[str, Any]]]]:
    callbacks: list[dict[str, Any]] = []

    def callback_handler(request: httpx.Request) -> httpx.Response:
        callbacks.append(json.loads(request.content))
        return httpx.Response(200)

    callback_client = httpx.AsyncClient(
        transport=httpx.MockTransport(callback_handler)
    )
    settings = make_settings(tmp_path)
    application = create_app(settings, FakeUpscaler(), callback_client)
    with TestClient(application) as client:
        yield client, callbacks


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "public_base_url": "https://upscale.example.com",
        "weights_dir": tmp_path / "weights",
        "upload_dir": tmp_path / "uploads",
        "result_dir": tmp_path / "results",
        "log_dir": tmp_path / "logs",
        "callback_attempts": 1,
        "max_upload_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)
