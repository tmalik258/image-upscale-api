"""
Inference harness for Real-ESRGAN and HAT via Spandrel.
Usage:
    uv run --no-sync python esrgan_test.py 4xl.png --crop 230 1835 580 1925 --name label
    uv run --no-sync python esrgan_test.py 4xl.png --tile 400 --name fullsheet
    uv run --no-sync python esrgan_test.py 4xl.png --model hat --name hat
"""

import argparse
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from app.api_schema.jobs import UpscaleModelType
from app.config import PROJECT_ROOT
from app.models.loader import WEIGHT_FILES, load_image_model


Image.MAX_IMAGE_PIXELS = None


def infer(model, tensor, device):
    with torch.inference_mode():
        output = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return output


def upscale(model, img, device, tile=0, pad=10):
    scale = model.scale
    array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, height, width = tensor.shape

    if tile <= 0:
        output = infer(model, tensor, device)
    else:
        canvas_device = "cpu" if device.type == "cuda" else device
        output = torch.zeros(
            1, 3, height * scale, width * scale, device=canvas_device
        )
        columns, rows = -(-width // tile), -(-height // tile)
        for row in range(rows):
            for column in range(columns):
                x0, y0 = column * tile, row * tile
                x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
                px0, py0 = max(x0 - pad, 0), max(y0 - pad, 0)
                px1, py1 = min(x1 + pad, width), min(y1 + pad, height)
                patch = infer(model, tensor[:, :, py0:py1, px0:px1], device)
                if canvas_device == "cpu":
                    patch = patch.cpu()
                ox, oy = (x0 - px0) * scale, (y0 - py0) * scale
                output[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = patch[
                    :, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale
                ]
                print(
                    f"  tile {row * columns + column + 1}/{columns * rows}",
                    end="\r",
                    flush=True,
                )
        print()

    rgb = output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((rgb * 255).round().astype(np.uint8)), scale


def selected_models(values: list[str] | None) -> list[UpscaleModelType]:
    if values is None:
        return list(UpscaleModelType)
    return [UpscaleModelType(value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument(
        "--model",
        action="append",
        choices=[item.value for item in UpscaleModelType],
        help="Repeat to run a subset. Default: both.",
    )
    parser.add_argument("--crop", nargs=4, type=int, metavar=("L", "T", "R", "B"))
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="400 recommended for full sheets",
    )
    parser.add_argument("--name", default="out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    image = Image.open(args.input).convert("RGB")
    if args.crop:
        image = image.crop(tuple(args.crop))
    width, height = image.size
    if args.tile == 0 and width * height > 500_000:
        print(
            "warning: large input with tile=0 will likely OOM. try --tile 400",
            file=sys.stderr,
        )

    model_results: list[tuple[str, Image.Image]] = []
    comparison_scale = None
    for model_type in selected_models(args.model):
        weights_path = PROJECT_ROOT / "weights" / WEIGHT_FILES[model_type]
        model = load_image_model(weights_path, device)
        print(f"input: {width}x{height} scale={model.scale} ({model_type.value})")
        started = time.time()
        upscaled, scale = upscale(model, image, device, tile=args.tile)
        print(f"{model_type.value}: {time.time() - started:.1f}s")
        upscaled.save(f"{args.name}_{model_type.value}.png")
        model_results.append((model_type.value.upper(), upscaled))
        comparison_scale = scale

    if comparison_scale is None:
        raise ValueError("No models were selected")

    nearest = image.resize(
        (width * comparison_scale, height * comparison_scale),
        Image.Resampling.NEAREST,
    )
    lanczos = image.resize(
        (width * comparison_scale, height * comparison_scale),
        Image.Resampling.LANCZOS,
    )
    nearest.save(f"{args.name}_orig.png")
    lanczos.save(f"{args.name}_lanczos.png")
    rows = [
        ("ORIGINAL (nearest)", nearest),
        ("LANCZOS", lanczos),
        *model_results,
    ]

    row_height = height * comparison_scale
    if row_height * len(rows) < 20000:
        padpx = 34
        canvas = Image.new(
            "RGB",
            (width * comparison_scale, (row_height + padpx) * len(rows)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, (label, rendered) in enumerate(rows):
            y = index * (row_height + padpx)
            draw.text((8, y + 9), label, fill="black")
            canvas.paste(rendered, (0, y + padpx))
        canvas.save(f"{args.name}_compare.png")
        print(f"wrote {args.name}_compare.png")


if __name__ == "__main__":
    main()
