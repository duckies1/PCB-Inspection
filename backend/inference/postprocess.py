"""Convert Ultralytics results into the stable API contract."""

from typing import Any

from backend.api.schemas import BoundingBox, Detection


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detections_from_result(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {})
    detections: list[Detection] = []
    for coordinates, confidence, class_tensor in zip(boxes.xyxy, boxes.conf, boxes.cls):
        class_id = int(class_tensor.item())
        values = [float(value) for value in coordinates.tolist()]
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(names, class_id),
                confidence=float(confidence.item()),
                bbox=BoundingBox(
                    x1=values[0], y1=values[1], x2=values[2], y2=values[3]
                ),
            )
        )
    return detections
