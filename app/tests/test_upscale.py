import json
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.conftest import FailingUpscaler, FakeUpscaler, make_settings


def wait_for_callback(
    client: TestClient,
    job_id: str,
    timeout: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/upscale/jobs/{job_id}")
        payload = response.json()
        if payload["callback_delivered"] or payload["callback_error"]:
            return payload
        time.sleep(0.01)
    raise AssertionError("Job callback did not finish")


def test_job_returns_upscaled_png_and_webhook_metadata(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
) -> None:
    client, callbacks = api_client
    response = client.post(
        "/api/v1/upscale/jobs",
        files={"image": ("source.png", image_bytes, "image/png")},
        data={"callback_url": "https://n8n.example.com/webhook/upscaled", "tile": "32"},
    )

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert "/api/v1/upscale/jobs/" in accepted["status_url"]

    job = wait_for_callback(client, accepted["job_id"])
    assert job["status"] == "completed"
    assert job["callback_delivered"] is True
    assert job["model_type"] == "esrgan"
    assert job["input_width"] == 8
    assert job["output_width"] == 32
    assert job["result_url"].endswith(f"{accepted['job_id']}.png")

    result = client.get(f"/api/v1/upscale/results/{accepted['job_id']}.png")
    assert result.status_code == 200
    assert result.headers["content-type"] == "image/png"
    assert callbacks[0]["job_id"] == accepted["job_id"]
    assert callbacks[0]["result_url"] == job["result_url"]
    assert callbacks[0]["model_type"] == "esrgan"


def test_job_echoes_hat_model_type(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
) -> None:
    client, callbacks = api_client
    response = client.post(
        "/api/v1/upscale/jobs",
        files={"image": ("source.png", image_bytes, "image/png")},
        data={
            "callback_url": "https://n8n.example.com/webhook/upscaled",
            "model_type": "hat",
        },
    )

    assert response.status_code == 202
    job = wait_for_callback(client, response.json()["job_id"])
    assert job["model_type"] == "hat"
    assert callbacks[0]["model_type"] == "hat"


def test_invalid_and_oversized_uploads_are_rejected(
    api_client: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = api_client
    invalid = client.post(
        "/api/v1/upscale/jobs",
        files={"image": ("fake.png", b"not an image", "image/png")},
        data={"callback_url": "https://n8n.example.com/webhook/upscaled"},
    )
    oversized = client.post(
        "/api/v1/upscale/jobs",
        files={
            "image": (
                "large.png",
                b"x" * (1024 * 1024 + 1),
                "image/png",
            )
        },
        data={"callback_url": "https://n8n.example.com/webhook/upscaled"},
    )

    assert invalid.status_code == 400
    assert invalid.json() == {"detail": "Image source is not a valid image"}
    assert oversized.status_code == 413


def test_failed_inference_is_reported_to_webhook(
    tmp_path: Path,
    image_bytes: bytes,
) -> None:
    callbacks: list[dict[str, Any]] = []

    def callback_handler(request: httpx.Request) -> httpx.Response:
        callbacks.append(json.loads(request.content))
        return httpx.Response(200)

    callback_client = httpx.AsyncClient(
        transport=httpx.MockTransport(callback_handler)
    )
    application = create_app(
        make_settings(tmp_path),
        FailingUpscaler(),
        callback_client,
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/upscale/jobs",
            files={"image": ("source.png", image_bytes, "image/png")},
            data={"callback_url": "https://n8n.example.com/webhook/upscaled"},
        )
        job = wait_for_callback(client, response.json()["job_id"])

    assert job["status"] == "failed"
    assert job["error"] == "Test inference failure"
    assert job["result_url"] is None
    assert callbacks[0]["status"] == "failed"
    assert callbacks[0]["error"] == "Test inference failure"


def test_webhook_failure_is_exposed_in_job_metadata(
    tmp_path: Path,
    image_bytes: bytes,
) -> None:
    callback_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    )
    application = create_app(
        make_settings(tmp_path),
        FakeUpscaler(),
        callback_client,
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/upscale/jobs",
            files={"image": ("source.png", image_bytes, "image/png")},
            data={"callback_url": "https://n8n.example.com/webhook/upscaled"},
        )
        job = wait_for_callback(client, response.json()["job_id"])

    assert job["status"] == "completed"
    assert job["callback_delivered"] is False
    assert "503 Service Unavailable" in job["callback_error"]
