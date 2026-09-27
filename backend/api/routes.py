"""HTTP routes for PCB inference."""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.api.schemas import DetectionResponse, ErrorResponse, ImageInfo, InferenceInfo
from backend.config import ALLOWED_IMAGE_TYPES
from backend.inference.preprocess import InvalidImageError, decode_image

router = APIRouter(prefix="/api")


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect(request: Request, image: UploadFile = File(...)) -> DetectionResponse:
    settings = request.app.state.settings
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_IMAGE", "message": "The uploaded file is not a supported image."},
        )

    contents = await image.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "FILE_TOO_LARGE", "message": "The uploaded image exceeds the size limit."},
        )

    try:
        decoded = decode_image(contents, image.content_type, ALLOWED_IMAGE_TYPES)
        detections, time_ms, device = request.app.state.detector.predict(decoded)
    except InvalidImageError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_IMAGE", "message": str(error)},
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INFERENCE_FAILED", "message": "Inspection failed. Please try again."},
        ) from error

    return DetectionResponse(
        image=ImageInfo(width=decoded.width, height=decoded.height),
        classes=request.app.state.detector.class_names,
        detections=detections,
        inference=InferenceInfo(
            model=settings.model_name,
            time_ms=round(time_ms, 2),
            device=device,
        ),
    )
