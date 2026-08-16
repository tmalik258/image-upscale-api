from pathlib import Path

import torch
from spandrel import ImageModelDescriptor, ModelLoader

from app.api_schema.jobs import UpscaleModelType


WEIGHT_FILES: dict[UpscaleModelType, str] = {
    UpscaleModelType.ESRGAN: "RealESRGAN_x4plus.pth",
    UpscaleModelType.HAT: "Real_HAT_GAN_SRx4.pth",
}


def load_image_model(
    weights_path: Path,
    device: torch.device,
) -> ImageModelDescriptor:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    loaded = ModelLoader(device=device).load_from_file(str(weights_path))
    if not isinstance(loaded, ImageModelDescriptor):
        raise TypeError(
            f"Expected ImageModelDescriptor from {weights_path}, got {type(loaded).__name__}"
        )
    return loaded.eval()
