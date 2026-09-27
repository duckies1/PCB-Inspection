"""Environment-backed application configuration."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(
        os.getenv("MODEL_PATH", str(PROJECT_ROOT / "runs/detect/runs/pku_yolo26s-3/weights/best.pt"))
    )
    model_name: str = os.getenv("MODEL_NAME", "YOLO26s")
    device: str | None = os.getenv("DEVICE") or None
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(20 * 1024 * 1024)))
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    confidence: float = float(os.getenv("MODEL_CONFIDENCE", "0.25"))


settings = Settings()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
