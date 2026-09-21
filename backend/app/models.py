"""
SQLAlchemy ORM models.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Match ─────────────────────────────────────────────────────────────────────

class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    video_path: Mapped[str] = mapped_column(String(1024))
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Processing state
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    # Possible: uploaded | queued | processing | completed | failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_fps: Mapped[int] = mapped_column(Integer, default=5)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Calibration
    homography_matrix: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON-serialized 3×3 matrix
    calibration_points: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON image→pitch point pairs

    # Relations
    teams: Mapped[list["Team"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    players: Mapped[list["Player"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    tactical_events: Mapped[list["TacticalEvent"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


# ── Team ──────────────────────────────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(16))  # "team_a" | "team_b" | "referee"
    display_name: Mapped[str] = mapped_column(String(64), default="")
    color_hex: Mapped[str] = mapped_column(String(7), default="#ffffff")

    match: Mapped["Match"] = relationship(back_populates="teams")
    players: Mapped[list["Player"]] = relationship(back_populates="team")


# ── Player ────────────────────────────────────────────────────────────────────

class Player(Base):
    __tablename__ = "players"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    team_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )
    tracking_id: Mapped[int] = mapped_column(Integer)  # ByteTrack persistent ID
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_goalkeeper: Mapped[bool] = mapped_column(Boolean, default=False)
    is_referee: Mapped[bool] = mapped_column(Boolean, default=False)

    # Computed analytics (stored after full processing)
    total_distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pct_defensive_third: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pct_middle_third: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pct_attacking_third: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="players")
    team: Mapped[Optional["Team"]] = relationship(back_populates="players")
    tracking_points: Mapped[list["TrackingPoint"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )

    @property
    def team_label(self) -> Optional[str]:
        return self.team.label if self.team else None

    @property
    def team_color(self) -> Optional[str]:
        return self.team.color_hex if self.team else None


# ── TrackingPoint ──────────────────────────────────────────────────────────────

class TrackingPoint(Base):
    __tablename__ = "tracking_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"))
    frame: Mapped[int] = mapped_column(Integer, index=True)
    timestamp_s: Mapped[float] = mapped_column(Float)

    # Image coordinates (pixels)
    img_x: Mapped[float] = mapped_column(Float)
    img_y: Mapped[float] = mapped_column(Float)

    # Pitch coordinates (metres, after homography)
    pitch_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Speed in km/h at this frame (estimated)
    speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    player: Mapped["Player"] = relationship(back_populates="tracking_points")


# ── BallTrackingPoint ──────────────────────────────────────────────────────────

class BallTrackingPoint(Base):
    __tablename__ = "ball_tracking_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    frame: Mapped[int] = mapped_column(Integer, index=True)
    timestamp_s: Mapped[float] = mapped_column(Float)
    img_x: Mapped[float] = mapped_column(Float)
    img_y: Mapped[float] = mapped_column(Float)
    pitch_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


# ── TacticalEvent ─────────────────────────────────────────────────────────────

class TacticalEvent(Base):
    __tablename__ = "tactical_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    timestamp_s: Mapped[float] = mapped_column(Float)
    frame: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(64))
    # e.g. "formation_change" | "tactical_phase" | "possession_change"
    team_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    value: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="tactical_events")
