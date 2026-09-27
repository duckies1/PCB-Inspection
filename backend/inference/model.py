"""Replaceable YOLO-backed detector implementation."""

from time import perf_counter
from typing import Any

from PIL import Image

from backend.api.schemas import Detection
from backend.config import Settings
from backend.inference.postprocess import detections_from_result


class PCBDetector:
    """Owns model loading and prediction; the API never depends on Ultralytics."""

    def __init__(self, settings: Settings) -> None:
        if not settings.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {settings.model_path}")

        from ultralytics import YOLO

        self.settings = settings
        self.model = YOLO(str(settings.model_path))
        self.device = settings.device or "auto"
        names = getattr(self.model, "names", {})
        self.class_names = [str(name) for _, name in sorted(names.items())] if isinstance(names, dict) else [str(name) for name in names]

    def predict(self, image: Image.Image) -> tuple[list[Detection], float, str]:
        started = perf_counter()
        results = self.model.predict(
            source=image,
            conf=self.settings.confidence,
            device=self.settings.device,
            verbose=False,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        result = results[0]
        return detections_from_result(result), elapsed_ms, self._device_name()

    def _device_name(self) -> str:
        if self.settings.device:
            return self.settings.device.upper()
        model_device = getattr(self.model, "device", None)
        return str(model_device or "AUTO").upper()
