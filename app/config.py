from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Image Upscaling API"
    public_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    weights_dir: Path = PROJECT_ROOT / "weights"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    result_dir: Path = PROJECT_ROOT / "data" / "results"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    default_tile: int = Field(default=400, ge=0)
    image_download_timeout_seconds: float = Field(default=30.0, gt=0)
    callback_timeout_seconds: float = Field(default=30.0, gt=0)
    callback_attempts: int = Field(default=3, ge=1, le=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def prepare_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def absolute_url(self, path: str) -> str:
        return f"{str(self.public_base_url).rstrip('/')}/{path.lstrip('/')}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
