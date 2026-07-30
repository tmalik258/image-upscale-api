# Image Upscaling API

Asynchronous 4x image upscaling with Real-ESRGAN. The API accepts an image,
processes one job at a time, and sends the result URL and metadata to an n8n
webhook.

## Setup

Create a Python virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `PUBLIC_BASE_URL` to an address that both
n8n and its users can reach. A localhost URL only works when n8n runs on the
same machine or network.

The default model file is `weights/RealESRGAN_x4plus.pth`.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Submit an upscaling job

Send a multipart request from the n8n HTTP Request node. Provide exactly one
of `image` or `image_url`.

### Upload an image file

```bash
curl -X POST http://localhost:8000/api/v1/upscale/jobs \
  -F "image=@4xl.png" \
  -F "callback_url=https://example.app/webhook/upscale-complete" \
  -F "tile=400"
```

### Download an image from a URL

```bash
curl -X POST http://localhost:8000/api/v1/upscale/jobs \
  -F "image_url=https://images.example.com/source.png" \
  -F "callback_url=https://example.app/webhook/upscale-complete" \
  -F "tile=400"
```

The remote URL must use HTTP or HTTPS, resolve to a public address, and return
JPEG, PNG, or WebP content. Downloads use the same `MAX_UPLOAD_BYTES` limit as
file uploads and the `IMAGE_DOWNLOAD_TIMEOUT_SECONDS` timeout.

Optional crop fields are `crop_left`, `crop_top`, `crop_right`, and
`crop_bottom`. All four are required when cropping.

The API immediately returns `202 Accepted`:

```json
{
  "job_id": "6679a328-e851-4cde-871d-29ca3711d7fb",
  "status": "queued",
  "status_url": "https://api.example.com/api/v1/upscale/jobs/6679a328-e851-4cde-871d-29ca3711d7fb"
}
```

When processing completes, the API sends JSON to `callback_url`:

```json
{
  "job_id": "6679a328-e851-4cde-871d-29ca3711d7fb",
  "status": "completed",
  "result_url": "https://api.example.com/api/v1/upscale/results/6679a328-e851-4cde-871d-29ca3711d7fb.png",
  "input_width": 512,
  "input_height": 512,
  "output_width": 2048,
  "output_height": 2048,
  "duration_seconds": 12.4,
  "completed_at": "2026-07-29T12:00:00Z",
  "error": null
}
```

Failed jobs use `status: "failed"`, omit `result_url`, and include `error`.
Callbacks are retried according to `CALLBACK_ATTEMPTS`.

## Job lifecycle

- `GET /api/v1/upscale/jobs/{job_id}` returns status and metadata.
- `GET /api/v1/upscale/results/{job_id}.png` downloads a completed PNG.
- Jobs are kept in memory and do not survive an API restart.
- Result PNG files remain in `data/results`.
- A single worker serializes model inference to control memory usage.

## Test

```bash
pytest
```
