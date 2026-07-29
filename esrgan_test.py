"""
Real-ESRGAN x4plus test harness -- COMMENTED VERSION.
Every line that does something non-obvious has a comment explaining
what it does AND why. Read this alongside LEARNING_NOTES.docx --
the document explains the concepts, this file shows where they
show up in actual code.
Setup (WSL Ubuntu):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install pillow numpy
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
Usage:
    python esrgan_test_commented.py 4xl.png --crop 230 1835 580 1925 --name label
    python esrgan_test_commented.py 4xl.png --tile 400 --name fullsheet
"""

import argparse, time, sys

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(nf, gc) for _ in range(3))

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_model(weights, device, nb=23):
    net = RRDBNet(nb=nb)
    sd = torch.load(weights, map_location="cpu")
    net.load_state_dict(sd["params_ema"])
    return net.eval().to(device)


def infer(net, t):
    with torch.no_grad():
        return net(t)


def upscale(net, img, device, tile=0, pad=10, scale=4):
    """Tiled forward pass. tile=0 processes the whole image at once."""

    a = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape

    if tile <= 0:
        out = infer(net, t)
    else:
        out = torch.zeros(1, 3, h * scale, w * scale, device=device)
        nx, ny = -(-w // tile), -(-h // tile)

        for yi in range(ny):
            for xi in range(nx):
                x0, y0 = xi * tile, yi * tile
                x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
                px0, py0 = max(x0 - pad, 0), max(y0 - pad, 0)
                px1, py1 = min(x1 + pad, w), min(y1 + pad, h)
                patch = infer(net, t[:, :, py0:py1, px0:px1])
                ox, oy = (x0 - px0) * scale, (y0 - py0) * scale
                out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = patch[
                    :, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale
                ]
                print(f"  tile {yi * nx + xi + 1}/{nx * ny}", end="\r", flush=True)

        print()

    o = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((o * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--weights", default="RealESRGAN_x4plus.pth")
    ap.add_argument(
        "--nb",
        type=int,
        default=23,
        help="23 for RealESRGAN_x4plus.pth, 6 for RealESRGAN_x4plus_anime_6B.pth",
    )
    ap.add_argument("--crop", nargs=4, type=int, metavar=("L", "T", "R", "B"))
    ap.add_argument(
        "--tile", type=int, default=0, help="400 recommended for full sheets"
    )
    ap.add_argument("--name", default="out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("note: ~2,000 px/s. A full 2167x1935 sheet takes ~34 min.")

    img = Image.open(args.input).convert("RGB")
    if args.crop:
        img = img.crop(tuple(args.crop))
    w, h = img.size
    print(f"input: {w}x{h} -> output {w * 4}x{h * 4} ({w * h * 16 / 1e6:.1f} MP)")

    if args.tile == 0 and w * h > 500_000:
        print(
            "warning: large input with tile=0 will likely OOM. try --tile 400",
            file=sys.stderr,
        )

    net = load_model(args.weights, device, nb=args.nb)
    t0 = time.time()
    esr = upscale(net, img, device, tile=args.tile)
    print(f"real-esrgan: {time.time() - t0:.1f}s")

    lan = img.resize((w * 4, h * 4), Image.LANCZOS)
    nn_ = img.resize((w * 4, h * 4), Image.NEAREST)

    for suffix, im in (("orig", nn_), ("lanczos", lan), ("esrgan", esr)):
        im.save(f"{args.name}_{suffix}.png")

    if h * 4 * 3 < 20000:
        from PIL import ImageDraw

        padpx = 34
        canvas = Image.new("RGB", (w * 4, (h * 4 + padpx) * 3), "white")
        d = ImageDraw.Draw(canvas)

        for i, (lbl, im) in enumerate(
            [("ORIGINAL (nearest)", nn_), ("LANCZOS 4x", lan), ("REAL-ESRGAN 4x", esr)]
        ):
            y = i * (h * 4 + padpx)
            d.text((8, y + 9), lbl, fill="black")
            canvas.paste(im, (0, y + padpx))
        canvas.save(f"{args.name}_compare.png")
        print(f"wrote {args.name}_compare.png")


if __name__ == "__main__":
    main()
