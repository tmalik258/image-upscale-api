import asyncio
import ipaddress
import socket
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile, status
from PIL import Image


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
CHUNK_SIZE = 1024 * 1024
MAX_REDIRECTS = 3


async def save_upload(
    upload: UploadFile,
    destination: Path,
    maximum_bytes: int,
) -> None:
    total_bytes = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > maximum_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Image exceeds the {maximum_bytes}-byte upload limit",
                )
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty",
        )


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


async def download_image(
    image_url: str,
    destination: Path,
    maximum_bytes: int,
    timeout_seconds: float,
) -> None:
    current_url = httpx.URL(image_url)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                await _validate_public_url(current_url)
                async with client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_REDIRECTS:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Image URL has too many or invalid redirects",
                            )
                        current_url = response.url.join(location)
                        continue

                    response.raise_for_status()
                    _validate_remote_response(response, maximum_bytes)
                    await _write_response(response, destination, maximum_bytes)
                    return
    except HTTPException:
        raise
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to download image from the supplied URL",
        ) from error


async def _validate_public_url(url: httpx.URL) -> None:
    if url.scheme not in {"http", "https"} or not url.host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image URL must use HTTP or HTTPS",
        )

    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            url.host,
            url.port or (443 if url.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image URL host could not be resolved",
        ) from error

    if any(
        not ipaddress.ip_address(address[4][0]).is_global
        for address in addresses
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image URL cannot target a private or reserved address",
        )


def _validate_remote_response(
    response: httpx.Response,
    maximum_bytes: int,
) -> None:
    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image URL must return JPEG, PNG, or WebP content",
        )

    content_length = response.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > maximum_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Image exceeds the {maximum_bytes}-byte download limit",
            )


async def _write_response(
    response: httpx.Response,
    destination: Path,
    maximum_bytes: int,
) -> None:
    total_bytes = 0
    with destination.open("wb") as output:
        async for chunk in response.aiter_bytes(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > maximum_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Image exceeds the {maximum_bytes}-byte download limit",
                )
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image URL returned an empty response",
        )
