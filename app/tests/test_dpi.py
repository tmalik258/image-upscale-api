from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from PIL import Image


SOURCE_URL = "https://images.example.com/source.png"


def stub_download(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    async def fake_download(
        image_url: str,
        destination: Path,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> None:
        assert image_url == SOURCE_URL
        assert maximum_bytes > 0
        assert timeout_seconds > 0
        destination.write_bytes(payload)

    monkeypatch.setattr("app.routes.dpi.download_image", fake_download)


def read_result(client: TestClient, result_url: str) -> tuple[bytes, str]:
    path = urlparse(result_url).path
    response = client.get(path)
    return response.content, response.headers["content-type"]


def assert_dpi(content: bytes, expected: int) -> None:
    with Image.open(BytesIO(content)) as image:
        horizontal, vertical = image.info["dpi"]
    assert round(horizontal) == expected
    assert round(vertical) == expected


def test_dpi_rewrite_defaults_to_jpeg_at_144(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client
    stub_download(monkeypatch, image_bytes)
    response = client.post("/api/v1/dpi", json={"image_url": SOURCE_URL})

    assert response.status_code == 200
    payload = response.json()
    assert payload["dpi"] == 144
    assert payload["format"] == "jpg"
    assert payload["width"] == 8
    assert payload["height"] == 6
    assert payload["result_url"].endswith(".jpg")

    content, media_type = read_result(client, payload["result_url"])
    assert media_type == "image/jpeg"
    assert_dpi(content, 144)


def test_dpi_rewrite_can_write_png(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client
    stub_download(monkeypatch, image_bytes)
    response = client.post(
        "/api/v1/dpi",
        json={"image_url": SOURCE_URL, "format": "png"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "png"
    assert payload["result_url"].endswith(".png")

    content, media_type = read_result(client, payload["result_url"])
    assert media_type == "image/png"
    assert_dpi(content, 144)


def test_dpi_rewrite_honors_requested_dpi(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    image_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client
    stub_download(monkeypatch, image_bytes)
    response = client.post(
        "/api/v1/dpi",
        json={"image_url": SOURCE_URL, "dpi": 300},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dpi"] == 300
    content, _ = read_result(client, payload["result_url"])
    assert_dpi(content, 300)


def test_invalid_dpi_and_format_are_rejected(
    api_client: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = api_client
    too_low = client.post("/api/v1/dpi", json={"image_url": SOURCE_URL, "dpi": 0})
    too_high = client.post("/api/v1/dpi", json={"image_url": SOURCE_URL, "dpi": 1201})
    bad_format = client.post(
        "/api/v1/dpi",
        json={"image_url": SOURCE_URL, "format": "gif"},
    )

    assert too_low.status_code == 422
    assert too_high.status_code == 422
    assert bad_format.status_code == 422


def test_invalid_image_bytes_are_rejected(
    api_client: tuple[TestClient, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = api_client
    stub_download(monkeypatch, b"not an image")
    response = client.post("/api/v1/dpi", json={"image_url": SOURCE_URL})

    assert response.status_code == 400
    assert response.json() == {"detail": "Image source is not a valid image"}
