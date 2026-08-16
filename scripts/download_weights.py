from pathlib import Path

import httpx


HAT_URL = "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "weights" / "Real_HAT_GAN_SRx4.pth"


def main() -> None:
    if not DESTINATION.parent.is_dir():
        raise FileNotFoundError(f"Weights directory not found: {DESTINATION.parent}")
    if DESTINATION.is_file():
        print(f"Already exists: {DESTINATION}")
        return

    partial = DESTINATION.with_suffix(".pth.partial")
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=120.0,
            headers={"User-Agent": "image-upscale-api-weight-download"},
        ) as client:
            with client.stream("GET", HAT_URL) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        partial.replace(DESTINATION)
    except (httpx.HTTPError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download HAT weights from {HAT_URL} to {DESTINATION}"
        ) from error
    print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
