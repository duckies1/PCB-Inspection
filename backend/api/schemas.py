"""Model-independent API schemas for PCB inspection results."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class ImageInfo(BaseModel):
    width: int
    height: int


class InferenceInfo(BaseModel):
    model: str
    time_ms: float
    device: str


class DetectionResponse(BaseModel):
    image: ImageInfo
    classes: list[str]
    detections: list[Detection]
    inference: InferenceInfo


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
