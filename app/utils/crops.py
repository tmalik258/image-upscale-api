from fastapi import HTTPException, status

from app.api_schema.jobs import CropBox


def build_crop(
    left: int | None,
    top: int | None,
    right: int | None,
    bottom: int | None,
) -> CropBox | None:
    coordinates = (left, top, right, bottom)
    if all(value is None for value in coordinates):
        return None
    if any(value is None for value in coordinates):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="All four crop coordinates are required",
        )

    assert left is not None
    assert top is not None
    assert right is not None
    assert bottom is not None
    return CropBox(left=left, top=top, right=right, bottom=bottom)
