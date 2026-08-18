from pydantic import AnyHttpUrl, BaseModel, Field

from app.utils.dpi import OUTPUT_DPI, DpiOutputFormat


class DpiRewriteRequest(BaseModel):
    image_url: AnyHttpUrl
    dpi: int = Field(default=OUTPUT_DPI, ge=1, le=1200)
    format: DpiOutputFormat = DpiOutputFormat.JPG


class DpiRewriteResponse(BaseModel):
    result_url: str
    dpi: int
    format: DpiOutputFormat
    width: int
    height: int
