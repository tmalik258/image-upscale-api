from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor

from app.api_schema.jobs import UpscaleModelType
from app.models.loader import WEIGHT_FILES, load_image_model


Image.MAX_IMAGE_PIXELS = None
OUTPUT_DPI = 144


@dataclass(frozen=True)
class UpscaleResult:
    input_width: int
    input_height: int
    output_width: int
    output_height: int


class ImageUpscaler:
    def __init__(self, weights_dir: Path) -> None:
        if not weights_dir.is_dir():
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
        self.weights_dir = weights_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, ImageModelDescriptor] = {}

    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
        model_type: UpscaleModelType = UpscaleModelType.ESRGAN,
    ) -> UpscaleResult:
        model = self._model(model_type)
        with Image.open(input_path) as source:
            image = source.convert("RGB")

        if crop is not None:
            self._validate_crop(crop, image.size)
            image = image.crop(crop)

        input_width, input_height = image.size
        output = self._upscale(model, image, tile)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path, format="PNG", dpi=(OUTPUT_DPI, OUTPUT_DPI))
        output_width, output_height = output.size
        return UpscaleResult(
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
        )

    def _model(self, model_type: UpscaleModelType) -> ImageModelDescriptor:
        key = model_type.value
        if key not in self._models:
            weights_path = self.weights_dir / WEIGHT_FILES[model_type]
            self._models[key] = load_image_model(weights_path, self.device)
        return self._models[key]

    def _upscale(
        self,
        model: ImageModelDescriptor,
        image: Image.Image,
        tile: int,
        pad: int = 10,
    ) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        _, _, height, width = tensor.shape

        if tile <= 0:
            output = self._infer(model, tensor)
        else:
            output = self._upscale_tiled(model, tensor, width, height, tile, pad)

        output_array = (
            output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        )
        return Image.fromarray((output_array * 255).round().astype(np.uint8))

    def _upscale_tiled(
        self,
        model: ImageModelDescriptor,
        tensor: torch.Tensor,
        width: int,
        height: int,
        tile: int,
        pad: int,
    ) -> torch.Tensor:
        scale = model.scale
        output = torch.zeros(
            1,
            3,
            height * scale,
            width * scale,
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
                    model,
                    tensor[
                        :,
                        :,
                        padded_y_start:padded_y_end,
                        padded_x_start:padded_x_end,
                    ],
                )
                self._copy_patch(
                    output,
                    patch,
                    (x_start, y_start, x_end, y_end),
                    (padded_x_start, padded_y_start),
                    scale,
                )

        return output

    def _infer(
        self,
        model: ImageModelDescriptor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return model(tensor)

    @staticmethod
    def _copy_patch(
        output: torch.Tensor,
        patch: torch.Tensor,
        bounds: tuple[int, int, int, int],
        padded_start: tuple[int, int],
        scale: int,
    ) -> None:
        x_start, y_start, x_end, y_end = bounds
        padded_x_start, padded_y_start = padded_start
        offset_x = (x_start - padded_x_start) * scale
        offset_y = (y_start - padded_y_start) * scale
        output[
            :,
            :,
            y_start * scale : y_end * scale,
            x_start * scale : x_end * scale,
        ] = patch[
            :,
            :,
            offset_y : offset_y + (y_end - y_start) * scale,
            offset_x : offset_x + (x_end - x_start) * scale,
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
