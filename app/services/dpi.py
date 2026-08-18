from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.utils.dpi import DpiOutputFormat, save_image_with_dpi


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class DpiRewriteResult:
    width: int
    height: int
    dpi: int
    output_format: DpiOutputFormat


def rewrite_image_dpi(
    input_path: Path,
    output_path: Path,
    dpi: int,
    output_format: DpiOutputFormat,
) -> DpiRewriteResult:
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    save_image_with_dpi(image, output_path, dpi, output_format)
    width, height = image.size
    return DpiRewriteResult(
        width=width,
        height=height,
        dpi=dpi,
        output_format=output_format,
    )
