"""
Application configuration using Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "EPL AI Tactical Analyst"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/epl_analyst.db"

    # ── Storage paths ─────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    videos_dir: Path = Path("./data/videos")
    results_dir: Path = Path("./data/results")
    models_dir: Path = Path("./models")

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = 2048  # 2 GB

    # ── Processing defaults ───────────────────────────────────────────────────
    default_sample_fps: int = 5          # frames per second to analyze
    detection_confidence: float = 0.25
    tracking_confidence: float = 0.5
    yolo_model: str = "yolov8n.pt"       # can be swapped to a custom model

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:3000",
    ]

    # ── Pitch standard dimensions (metres) ────────────────────────────────────
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    # ── Team clustering ───────────────────────────────────────────────────────
    team_cluster_k: int = 2              # number of team clusters

    # ── Temporal model ───────────────────────────────────────────────────────
    temporal_model_path: Path = Path("./models/temporal_transformer.pt")
    sequence_length: int = 20            # frames in one temporal window

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        for d in [
            self.data_dir,
            self.videos_dir,
            self.results_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
