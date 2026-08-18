import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import UUID, uuid4

import httpx

from app.api_schema.jobs import (
    CropBox,
    JobAccepted,
    JobRecord,
    JobResponse,
    JobStatus,
    UpscaleModelType,
    WebhookPayload,
    utc_now,
)
from app.config import Settings
from app.services.upscaler import UpscaleResult


logger = logging.getLogger(__name__)


class Upscaler(Protocol):
    def process(
        self,
        input_path: Path,
        output_path: Path,
        tile: int,
        crop: tuple[int, int, int, int] | None = None,
        model_type: UpscaleModelType = UpscaleModelType.ESRGAN,
        job_id: UUID | None = None,
    ) -> UpscaleResult: ...


class UpscalingJobService:
    def __init__(
        self,
        settings: Settings,
        upscaler: Upscaler,
        callback_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.upscaler = upscaler
        self.jobs: dict[UUID, JobRecord] = {}
        self.queue: asyncio.Queue[UUID] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.callback_client = callback_client or httpx.AsyncClient(
            timeout=settings.callback_timeout_seconds
        )
        self.owns_callback_client = callback_client is None

    def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(
                self._worker(),
                name="image-upscaling-worker",
            )

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
            self.worker_task = None
        if self.owns_callback_client:
            await self.callback_client.aclose()

    async def submit(
        self,
        input_path: Path,
        callback_url: str | None,
        tile: int,
        crop: CropBox | None,
        model_type: UpscaleModelType,
    ) -> JobAccepted:
        job_id = uuid4()
        record = JobRecord(
            job_id=job_id,
            callback_url=callback_url,
            input_path=input_path,
            result_path=self.settings.result_dir / f"{job_id}.png",
            tile=tile,
            crop=crop,
            model_type=model_type,
        )
        self.jobs[job_id] = record
        await self.queue.put(job_id)
        crop_label = "no" if crop is None else f"{crop.left},{crop.top},{crop.right},{crop.bottom}"
        logger.info(
            "Job %s queued model=%s tile=%s crop=%s queue_size=%s",
            job_id,
            model_type.value,
            tile,
            crop_label,
            self.queue.qsize(),
        )
        return JobAccepted(
            job_id=job_id,
            status=record.status,
            status_url=self._status_url(job_id),
        )

    def get(self, job_id: UUID) -> JobRecord | None:
        return self.jobs.get(job_id)

    def response_for(self, record: JobRecord) -> JobResponse:
        return JobResponse(
            **record.model_dump(
                exclude={"callback_url", "input_path", "result_path", "tile", "crop"}
            ),
            status_url=self._status_url(record.job_id),
        )

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._process_job(self.jobs[job_id])
            except Exception:
                logger.exception("Unexpected worker failure for job %s", job_id)
            finally:
                self.queue.task_done()

    async def _process_job(self, record: JobRecord) -> None:
        record.status = JobStatus.PROCESSING
        record.started_at = utc_now()
        started = perf_counter()
        logger.info(
            "Job %s processing started model=%s tile=%s",
            record.job_id,
            record.model_type.value,
            record.tile,
        )
        try:
            result = await asyncio.to_thread(
                self.upscaler.process,
                record.input_path,
                record.result_path,
                record.tile,
                record.crop.as_tuple() if record.crop else None,
                record.model_type,
                record.job_id,
            )
            record.duration_seconds = round(perf_counter() - started, 3)
            record.input_width = result.input_width
            record.input_height = result.input_height
            record.output_width = result.output_width
            record.output_height = result.output_height
            record.result_url = self._result_url(record.job_id)
            record.status = JobStatus.COMPLETED
            logger.info(
                "Job %s completed %sx%s -> %sx%s in %.1fs",
                record.job_id,
                result.input_width,
                result.input_height,
                result.output_width,
                result.output_height,
                record.duration_seconds,
            )
        except Exception as error:
            record.duration_seconds = round(perf_counter() - started, 3)
            record.status = JobStatus.FAILED
            record.error = str(error)
            logger.info(
                "Job %s failed after %.1fs: %s",
                record.job_id,
                record.duration_seconds,
                error,
            )
            logger.exception("Upscaling failed for job %s", record.job_id)
        finally:
            record.completed_at = utc_now()
            record.input_path.unlink(missing_ok=True)

        await self._send_callback(record)

    async def _send_callback(self, record: JobRecord) -> None:
        if record.callback_url is None:
            return

        payload = WebhookPayload(
            job_id=record.job_id,
            status=record.status,
            result_url=record.result_url,
            input_width=record.input_width,
            input_height=record.input_height,
            output_width=record.output_width,
            output_height=record.output_height,
            duration_seconds=record.duration_seconds,
            completed_at=record.completed_at or utc_now(),
            error=record.error,
            model_type=record.model_type,
        )
        callback_url = str(record.callback_url)

        for attempt in range(1, self.settings.callback_attempts + 1):
            try:
                response = await self.callback_client.post(
                    callback_url,
                    json=payload.model_dump(mode="json"),
                )
                response.raise_for_status()
                record.callback_delivered = True
                record.callback_error = None
                logger.info("Job %s webhook delivered", record.job_id)
                return
            except httpx.HTTPError as error:
                record.callback_error = str(error)
                logger.warning(
                    "Webhook attempt %s/%s failed for job %s: %s",
                    attempt,
                    self.settings.callback_attempts,
                    record.job_id,
                    error,
                )
                if attempt < self.settings.callback_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))

    def _status_url(self, job_id: UUID) -> str:
        return self.settings.absolute_url(f"/api/v1/upscale/jobs/{job_id}")

    def _result_url(self, job_id: UUID) -> str:
        return self.settings.absolute_url(f"/api/v1/upscale/results/{job_id}.png")
