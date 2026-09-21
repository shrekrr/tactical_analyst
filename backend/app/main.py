import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# Fix PyTorch 2.6+ weights_only issue with ultralytics checkpoints
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.database import init_db

# Import routers
from app.api import upload, analysis, matches, players, websocket as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting EPL Tactical Analyst API v{}", settings.app_version)
    settings.ensure_dirs()
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered football tactical analysis platform.",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api/matches", tags=["Upload"])
app.include_router(analysis.router, prefix="/api/matches", tags=["Analysis"])
app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
app.include_router(players.router, prefix="/api/matches", tags=["Players"])
app.include_router(ws_router.router, prefix="/ws", tags=["WebSocket"])

# ── Serve uploaded videos (for playback) ─────────────────────────────────────
app.mount(
    "/videos",
    StaticFiles(directory=str(settings.videos_dir)),
    name="videos",
)
app.mount(
    "/results",
    StaticFiles(directory=str(settings.results_dir)),
    name="results",
)


@app.get("/api/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    return {
        "status": "ok",
        "version": settings.app_version,
        "compute_device": device,
        "gpu": gpu_name,
    }
