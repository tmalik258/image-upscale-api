from enum import StrEnum
from pathlib import Path

from PIL import Image


OUTPUT_DPI = 144
JPEG_QUALITY = 100
JPEG_SUBSAMPLING = 0


class DpiOutputFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"


PILLOW_SAVE_FORMATS = {
    DpiOutputFormat.JPG: "JPEG",
    DpiOutputFormat.PNG: "PNG",
}

MEDIA_TYPES = {
    DpiOutputFormat.JPG: "image/jpeg",
    DpiOutputFormat.PNG: "image/png",
}


def save_image_with_dpi(
    image: Image.Image,
    output_path: Path,
    dpi: int = OUTPUT_DPI,
    output_format: DpiOutputFormat = DpiOutputFormat.PNG,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {
        "format": PILLOW_SAVE_FORMATS[output_format],
        "dpi": (dpi, dpi),
    }
    if output_format is DpiOutputFormat.JPG:
        save_kwargs["quality"] = JPEG_QUALITY
        save_kwargs["subsampling"] = JPEG_SUBSAMPLING
    image.save(output_path, **save_kwargs)
