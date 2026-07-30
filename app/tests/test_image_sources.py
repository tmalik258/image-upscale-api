import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


def wait_for_callback(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get(f"/api/v1/upscale/jobs/{job_id}").json()
        if payload["callback_delivered"] or payload["callback_error"]:
            return payload
        time.sleep(0.01)
    raise AssertionError("Job callback did not finish")


def test_job_accepts_image_url(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client

    async def fake_download(
        image_url: str,
        destination: Path,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> None:
        assert image_url == "https://images.example.com/source.png"
        assert maximum_bytes > len(image_bytes)
        assert timeout_seconds > 0
        destination.write_bytes(image_bytes)

    monkeypatch.setattr("app.routes.upscale.download_image", fake_download)
    response = client.post(
        "/api/v1/upscale/jobs",
        data={
            "callback_url": "https://n8n.example.com/webhook/upscaled",
            "image_url": "https://images.example.com/source.png",
        },
    )

    assert response.status_code == 202
    job = wait_for_callback(client, response.json()["job_id"])
    assert job["status"] == "completed"
    assert job["input_width"] == 8
    assert job["output_width"] == 32


def test_job_accepts_image_url_without_callback_url(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client

    async def fake_download(
        image_url: str,
        destination: Path,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> None:
        destination.write_bytes(image_bytes)

    monkeypatch.setattr("app.routes.upscale.download_image", fake_download)
    response = client.post(
        "/api/v1/upscale/jobs",
        data={
            "image_url": "https://images.example.com/source.png",
            "tile": "400",
        },
    )

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"


def test_job_requires_exactly_one_image_source(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
) -> None:
    client, _ = api_client
    callback = "https://n8n.example.com/webhook/upscaled"
    missing = client.post(
        "/api/v1/upscale/jobs",
        data={"callback_url": callback},
    )
    both = client.post(
        "/api/v1/upscale/jobs",
        files={"image": ("source.png", image_bytes, "image/png")},
        data={
            "callback_url": callback,
            "image_url": "https://images.example.com/source.png",
        },
    )

    expected = {"detail": "Provide exactly one of image or image_url"}
    assert missing.status_code == 422
    assert missing.json() == expected
    assert both.status_code == 422
    assert both.json() == expected
