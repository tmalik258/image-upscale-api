from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.models.esrgan import RRDBNet, load_esrgan_model


Image.MAX_IMAGE_PIXELS = None
UPSCALE_FACTOR = 4
OUTPUT_DPI = 144


@dataclass(frozen=True)
class UpscaleResult:
    input_width: int
    input_height: int
    output_width: int
    output_height: int


class ImageUpscaler:
    def __init__(self, weights_path: Path, blocks: int = 23) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RRDBNet = load_esrgan_model(weights_path, self.device, blocks)

    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
    ) -> UpscaleResult:
        with Image.open(input_path) as source:
            image = source.convert("RGB")

        if crop is not None:
            self._validate_crop(crop, image.size)
            image = image.crop(crop)

        input_width, input_height = image.size
        output = self._upscale(image, tile)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path, format="PNG", dpi=(OUTPUT_DPI, OUTPUT_DPI))
        output_width, output_height = output.size
        return UpscaleResult(
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )

    def _upscale(self, image: Image.Image, tile: int, pad: int = 10) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        _, _, height, width = tensor.shape

        if tile <= 0:
            output = self._infer(tensor)
        else:
            output = self._upscale_tiled(tensor, width, height, tile, pad)

        output_array = (
            output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        )
        return Image.fromarray((output_array * 255).round().astype(np.uint8))

    def _upscale_tiled(
        self,
        tensor: torch.Tensor,
        width: int,
        height: int,
        tile: int,
        pad: int,
    ) -> torch.Tensor:
        output = torch.zeros(
            1,
            3,
            height * UPSCALE_FACTOR,
            width * UPSCALE_FACTOR,
            device=self.device,
        )
        horizontal_tiles = -(-width // tile)
        vertical_tiles = -(-height // tile)

        for tile_y in range(vertical_tiles):
            for tile_x in range(horizontal_tiles):
                x_start, y_start = tile_x * tile, tile_y * tile
                x_end = min(x_start + tile, width)
                y_end = min(y_start + tile, height)
                padded_x_start, padded_y_start = max(x_start - pad, 0), max(
                    y_start - pad, 0
                )
                padded_x_end = min(x_end + pad, width)
                padded_y_end = min(y_end + pad, height)
                patch = self._infer(
                    tensor[
                        :,
                        :,
                        padded_y_start:padded_y_end,
                        padded_x_start:padded_x_end,
                    ]
                )
                self._copy_patch(
                    output,
                    patch,
                    (x_start, y_start, x_end, y_end),
                    (padded_x_start, padded_y_start),
                )

        return output

    def _infer(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(tensor)

    @staticmethod
    def _copy_patch(
        output: torch.Tensor,
        patch: torch.Tensor,
        bounds: tuple[int, int, int, int],
        padded_start: tuple[int, int],
    ) -> None:
        x_start, y_start, x_end, y_end = bounds
        padded_x_start, padded_y_start = padded_start
        offset_x = (x_start - padded_x_start) * UPSCALE_FACTOR
        offset_y = (y_start - padded_y_start) * UPSCALE_FACTOR
        output[:, :, y_start * 4 : y_end * 4, x_start * 4 : x_end * 4] = patch[
            :,
            :,
            offset_y : offset_y + (y_end - y_start) * UPSCALE_FACTOR,
            offset_x : offset_x + (x_end - x_start) * UPSCALE_FACTOR,
        ]

    @staticmethod
    def _validate_crop(
        crop: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> None:
        left, top, right, bottom = crop
        width, height = image_size
        if left >= right or top >= bottom or right > width or bottom > height:
            raise ValueError("Crop must be ordered and contained within the image")
