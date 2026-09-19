"""
Pydantic request/response schemas.
"""
from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ── Match schemas ─────────────────────────────────────────────────────────────

class MatchUploadResponse(BaseModel):
    match_id: str
    status: str
    filename: str
    file_size_bytes: int


class MatchStatusResponse(BaseModel):
    match_id: str
    status: str
    progress: int
    error_message: Optional[str] = None


class MatchInfoResponse(BaseModel):
    id: str
    filename: str
    duration_s: Optional[float]
    fps: Optional[float]
    total_frames: Optional[int]
    width: Optional[int]
    height: Optional[int]
    file_size_bytes: Optional[int]
    status: str
    progress: int
    sample_fps: int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Calibration schemas ───────────────────────────────────────────────────────

class CalibrationPoint(BaseModel):
    """A single image→pitch calibration correspondence."""
    img_x: float
    img_y: float
    pitch_x: float  # metres from top-left of pitch
    pitch_y: float  # metres from top-left of pitch


class CalibrationRequest(BaseModel):
    points: list[CalibrationPoint] = Field(
        ..., min_length=4, description="At least 4 point correspondences"
    )


class CalibrationResponse(BaseModel):
    success: bool
    message: str
    homography_matrix: Optional[list[list[float]]] = None


# ── Player schemas ────────────────────────────────────────────────────────────

class PlayerSummary(BaseModel):
    id: str
    tracking_id: int
    team_label: Optional[str]
    team_color: Optional[str]
    total_distance_m: Optional[float]
    avg_speed_kmh: Optional[float]
    max_speed_kmh: Optional[float]
    avg_x: Optional[float]
    avg_y: Optional[float]
    pct_defensive_third: Optional[float]
    pct_middle_third: Optional[float]
    pct_attacking_third: Optional[float]
    is_goalkeeper: bool
    is_referee: bool

    class Config:
        from_attributes = True


class TrackingPointOut(BaseModel):
    frame: int
    timestamp_s: float
    pitch_x: Optional[float]
    pitch_y: Optional[float]
    speed_kmh: Optional[float]

    class Config:
        from_attributes = True


class PlayerDetailResponse(BaseModel):
    player: PlayerSummary
    trajectory: list[TrackingPointOut]
    heatmap_data: list[list[float]]  # 2D grid values
    speed_series: list[dict]  # [{timestamp_s, speed_kmh}]


# ── Team schemas ──────────────────────────────────────────────────────────────

class TeamMetrics(BaseModel):
    team_label: str
    display_name: str
    color_hex: str
    avg_width_m: Optional[float]
    avg_depth_m: Optional[float]
    avg_defensive_line_m: Optional[float]
    avg_compactness_m: Optional[float]
    total_distance_m: Optional[float]
    estimated_possession_pct: Optional[float]
    current_formation: Optional[str]
    heatmap_grid: Optional[List[List[float]]] = None  # rows x cols, values in [0,1]


# ── Tactical schemas ──────────────────────────────────────────────────────────

class TacticalEventOut(BaseModel):
    id: str
    timestamp_s: float
    frame: int
    event_type: str
    team_label: Optional[str]
    value: Optional[str]
    confidence: Optional[float]

    class Config:
        from_attributes = True


class FrameSnapshot(BaseModel):
    """All player/ball positions for a single frame."""
    frame: int
    timestamp_s: float
    players: list[dict]  # [{player_id, tracking_id, pitch_x, pitch_y, team_label}]
    ball: Optional[dict]  # {pitch_x, pitch_y}
    formation_a: Optional[str]
    formation_b: Optional[str]
    tactical_phase: Optional[str]
    possession: Optional[str]


class MatchAnalyticsResponse(BaseModel):
    match_id: str
    teams: list[TeamMetrics]
    players: list[PlayerSummary]
    tactical_events: list[TacticalEventOut]
    possession_timeline: list[dict]  # [{timestamp_s, team_label}]
    formation_timeline: list[dict]   # [{timestamp_s, team_label, formation}]
    summary: dict[str, Any]          # tactical summary text


# ── Analysis request ──────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    sample_fps: int = Field(default=5, ge=1, le=25)
    yolo_model: Optional[str] = None


# ── Progress ──────────────────────────────────────────────────────────────────

class ProgressMessage(BaseModel):
    match_id: str
    stage: str
    progress: int
    message: str
