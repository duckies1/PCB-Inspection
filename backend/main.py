"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.config import settings
from backend.inference.model import PCBDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.detector = PCBDetector(settings)
    yield


app = FastAPI(title="PCB Inspection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model": request.app.state.settings.model_name}


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, error: HTTPException) -> JSONResponse:
    detail = error.detail if isinstance(error.detail, dict) else {
        "code": "REQUEST_ERROR",
        "message": str(error.detail),
    }
    return JSONResponse(status_code=error.status_code, content={"error": detail})


@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, _error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "SERVER_ERROR", "message": "The inspection server encountered an error."}},
    )
