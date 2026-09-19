"""
Main processing pipeline.

Orchestrates the full CV→Analytics pipeline for a single match.
Designed to run in a subprocess (ProcessPoolExecutor) so it does not
block the FastAPI event loop.

Stages
------
1.  Video frame extraction + sampling
2.  YOLO + ByteTrack (detection + tracking per sampled frame)
3.  Team colour clustering → team assignment
4.  Homography transform → pitch coordinates
5.  Persist TrackingPoints to DB
6.  Compute player-level analytics
7.  Compute formation + tactical events
8.  Compute possession timeline
9.  Mark match as completed

Progress is updated in the DB at each stage (0–100%).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# These imports happen inside the subprocess — fresh Python interpreter
from app.config import settings
from app.analytics.formation import detect_formation
from app.analytics.player_metrics import (
    compute_average_position,
    compute_distance_covered,
    compute_speed_series,
    compute_zone_distribution,
)
from app.analytics.possession import nearest_team, smooth_possession
from app.analytics.tactical_phases import classify_phase
from app.detection.yolo_detector import YOLODetector
from app.models.temporal_model import TemporalFormationTransformer
from app.pitch.calibration import load_homography
from app.pitch.homography import HomographyTransformer
from app.teams.classifier import TeamClassifier
from app.tracking.byte_tracker import run_tracking_on_frame


# ── DB helpers (synchronous SQLAlchemy for subprocess) ────────────────────────

def _get_sync_db():
    """Return a synchronous SQLAlchemy session for use in subprocess."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_url = str(settings.database_url).replace("+aiosqlite", "")
    engine = create_engine(db_url)
    Session = sessionmaker(engine)
    return Session()


def _update_progress(session, match_id: str, progress: int, status: str = "processing") -> None:
    from app.models import Match
    match = session.get(Match, match_id)
    if match:
        match.progress = progress
        match.status = status
        session.commit()


def _set_error(session, match_id: str, message: str) -> None:
    from app.models import Match
    match = session.get(Match, match_id)
    if match:
        match.status = "failed"
        match.error_message = message
        session.commit()


# ── Main pipeline function ────────────────────────────────────────────────────

def run_pipeline(
    match_id: str,
    sample_fps: int = 5,
    yolo_model_path: str = "yolov8n.pt",
) -> None:
    """
    Full processing pipeline. Runs synchronously in a subprocess.

    Parameters
    ----------
    match_id : str
        UUID of the match record.
    sample_fps : int
        Frames per second to sample (default 5).
    yolo_model_path : str
        YOLO weight file path or ultralytics model name.
    """
    session = _get_sync_db()

    try:
        _run(session, match_id, sample_fps, yolo_model_path)
    except Exception as exc:
        logger.exception("Pipeline failed for match {}", match_id)
        _set_error(session, match_id, str(exc))
    finally:
        session.close()


def _run(session, match_id: str, sample_fps: int, yolo_model_path: str) -> None:
    from app.models import (
        BallTrackingPoint,
        Match,
        Player,
        TacticalEvent,
        Team,
        TrackingPoint,
    )

    # ── Load match ────────────────────────────────────────────────────────────
    match = session.get(Match, match_id)
    if not match:
        raise RuntimeError(f"Match {match_id} not found in DB")

    match.status = "processing"
    match.progress = 0
    session.commit()

    video_path = Path(match.video_path)
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(source_fps / sample_fps)))

    logger.info(
        "Pipeline start: match={} fps={}/{} step={} total={}",
        match_id, sample_fps, source_fps, frame_step, total_frames
    )

    # ── Initialise models ─────────────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Compute device: {}", device)

    detector = YOLODetector(
        model_path=yolo_model_path,
        confidence=settings.detection_confidence,
        device=device,
    )
    team_classifier = TeamClassifier(n_clusters=settings.team_cluster_k)

    # Homography transformer
    H = load_homography(match.homography_matrix)
    transformer = HomographyTransformer(
        H,
        pitch_length_m=settings.pitch_length_m,
        pitch_width_m=settings.pitch_width_m,
    )

    # ── Load trained formation classifier ─────────────────────────────────────
    formation_model = None
    _model_path = Path(__file__).parent.parent.parent.parent / "models" / "transformer" / "best_model.pt"
    if _model_path.exists():
        try:
            from app.models.temporal_model import TransformerFormationClassifier
            formation_model = TransformerFormationClassifier.load(str(_model_path))
            logger.info("Loaded trained Transformer formation classifier from {}", _model_path)
        except Exception as exc:
            logger.warning("Could not load formation model ({}), using rule-based fallback.", exc)
    else:
        logger.info("No trained formation model at {} — using rule-based detector.", _model_path)

    _update_progress(session, match_id, 5)

    # ── Stage 1: Detection + Tracking ─────────────────────────────────────────
    # {track_id: [(frame, img_x, img_y, bbox, confidence)]}
    track_data: Dict[int, List] = defaultdict(list)
    # Ball: [(frame, img_x, img_y, confidence)]
    ball_data: List[Tuple] = []
    # For team clustering: {track_id: [(frame, bbox)]}
    color_samples: Dict[int, List] = defaultdict(list)

    frame_idx = 0
    sampled = 0
    SAMPLE_COLOR_EVERY = 5  # take color sample every N sampled frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        tracks, ball = run_tracking_on_frame(
            detector.model,
            frame,
            frame_idx,
            tracker_cfg="bytetrack.yaml",
            confidence=settings.detection_confidence,
            device=device,
        )

        for track in tracks:
            track_data[track.track_id].append(
                (frame_idx, track.bottom_center_x, track.bottom_center_y, track.bbox, track.confidence)
            )
            if sampled % SAMPLE_COLOR_EVERY == 0:
                color_samples[track.track_id].append((frame, track.bbox))

        if ball:
            bx = (ball["bbox"][0] + ball["bbox"][2]) / 2.0
            by = ball["bbox"][3]  # bottom center
            ball_data.append((frame_idx, bx, by, ball["confidence"]))

        sampled += 1
        progress = int(5 + (frame_idx / max(total_frames, 1)) * 55)
        _update_progress(session, match_id, min(progress, 60))
        frame_idx += 1

    cap.release()
    logger.info("Detection done: {} tracks, {} ball points, {} sampled frames",
                len(track_data), len(ball_data), sampled)

    # ── Stage 2: Team classification ──────────────────────────────────────────
    _update_progress(session, match_id, 62)

    for tid, samples in color_samples.items():
        for frm, bbox in samples:
            team_classifier.add_sample(tid, frm, bbox)

    team_classifier.fit()

    # ── Stage 3: Create Team records ─────────────────────────────────────────
    team_a_color = team_classifier.get_color("team_a")
    team_b_color = team_classifier.get_color("team_b")

    team_a = Team(match_id=match_id, label="team_a", display_name="Team A", color_hex=team_a_color)
    team_b = Team(match_id=match_id, label="team_b", display_name="Team B", color_hex=team_b_color)
    ref_team = Team(match_id=match_id, label="referee", display_name="Referee", color_hex="#ffff00")
    session.add_all([team_a, team_b, ref_team])
    session.commit()

    team_map = {"team_a": team_a.id, "team_b": team_b.id, "referee": ref_team.id}

    # ── Stage 4: Create Player + TrackingPoint records ────────────────────────
    _update_progress(session, match_id, 65)

    player_db_map: Dict[int, str] = {}  # track_id → Player.id

    for track_id, points in track_data.items():
        team_label = team_classifier.get_team(track_id)
        if team_label == "unknown":
            team_label = "team_a"  # fallback

        player = Player(
            match_id=match_id,
            team_id=team_map.get(team_label),
            tracking_id=track_id,
            is_referee=(team_label == "referee"),
        )
        session.add(player)
        session.flush()
        player_db_map[track_id] = player.id

        # Insert tracking points in batch
        tp_objects = []
        for frame_no, img_x, img_y, bbox, conf in points:
            pitch_coord = transformer.transform(img_x, img_y) if transformer.is_calibrated else None
            tp_objects.append(
                TrackingPoint(
                    player_id=player.id,
                    frame=frame_no,
                    timestamp_s=frame_no / source_fps,
                    img_x=img_x,
                    img_y=img_y,
                    pitch_x=pitch_coord[0] if pitch_coord else None,
                    pitch_y=pitch_coord[1] if pitch_coord else None,
                    confidence=conf,
                )
            )
        session.bulk_save_objects(tp_objects)

    session.commit()

    # ── Stage 5: Ball tracking points ─────────────────────────────────────────
    if ball_data:
        ball_objects = []
        for frame_no, img_x, img_y, conf in ball_data:
            pitch_coord = transformer.transform(img_x, img_y) if transformer.is_calibrated else None
            ball_objects.append(
                BallTrackingPoint(
                    match_id=match_id,
                    frame=frame_no,
                    timestamp_s=frame_no / source_fps,
                    img_x=img_x,
                    img_y=img_y,
                    pitch_x=pitch_coord[0] if pitch_coord else None,
                    pitch_y=pitch_coord[1] if pitch_coord else None,
                    confidence=conf,
                )
            )
        session.bulk_save_objects(ball_objects)
        session.commit()

    _update_progress(session, match_id, 75)

    # ── Stage 6: Player analytics ─────────────────────────────────────────────
    for track_id, points in track_data.items():
        player_id = player_db_map.get(track_id)
        if not player_id:
            continue

        positions = []
        timestamps = []
        for frame_no, img_x, img_y, bbox, conf in sorted(points, key=lambda p: p[0]):
            if transformer.is_calibrated:
                coord = transformer.transform(img_x, img_y)
                if coord:
                    positions.append(coord)
                    timestamps.append(frame_no / source_fps)
            else:
                # Use normalised image coords as fallback
                positions.append((img_x, img_y))
                timestamps.append(frame_no / source_fps)

        if not positions:
            continue

        speeds = compute_speed_series(positions, timestamps)
        dist = compute_distance_covered(positions)
        avg_pos = compute_average_position(positions)
        zones = compute_zone_distribution(
            positions,
            pitch_length=settings.pitch_length_m if transformer.is_calibrated else 1920,
            pitch_width=settings.pitch_width_m if transformer.is_calibrated else 1080,
        )

        # Update speed in TrackingPoints
        player_obj = session.get(Player, player_id)
        if player_obj:
            player_obj.total_distance_m = dist
            player_obj.avg_speed_kmh = float(np.mean(speeds)) if speeds else None
            player_obj.max_speed_kmh = float(max(speeds)) if speeds else None
            player_obj.avg_x = avg_pos[0]
            player_obj.avg_y = avg_pos[1]
            player_obj.pct_defensive_third = zones.get("defensive_third")
            player_obj.pct_middle_third = zones.get("middle_third")
            player_obj.pct_attacking_third = zones.get("attacking_third")

        # Update speed on individual tracking points
        speed_map = {
            points[i][0]: speeds[i] for i in range(min(len(speeds), len(points)))
        }
        from sqlalchemy import select as sa_select
        tps = session.execute(
            sa_select(TrackingPoint).where(TrackingPoint.player_id == player_id)
        ).scalars().all()
        for tp in tps:
            tp.speed_kmh = speed_map.get(tp.frame)

    session.commit()
    _update_progress(session, match_id, 85)

    # ── Stage 7: Formation detection ──────────────────────────────────────────
    _detect_formations(session, match_id, match, source_fps, team_map, transformer, formation_model)
    _update_progress(session, match_id, 95)

    # ── Stage 8: Mark complete ─────────────────────────────────────────────────
    match_obj = session.get(Match, match_id)
    if match_obj:
        match_obj.status = "completed"
        match_obj.progress = 100
        session.commit()

    logger.info("Pipeline complete for match {}", match_id)


def _detect_formations(session, match_id, match, source_fps, team_map, transformer, formation_model=None):
    """Detect formations at regular intervals using trained model + rule-based fallback."""
    from sqlalchemy import select as sa_select
    from app.models import Player, Team, TacticalEvent, TrackingPoint

    WINDOW = 20          # frames per sequence (same as training)
    STEP = max(1, round(source_fps / 5))  # subsample to ~5fps (same as training)
    INTERVAL_S = 30      # report formation every 30 seconds
    INTERVAL_FRAMES = int(source_fps * INTERVAL_S)

    use_model = formation_model is not None
    N_PLAYERS = 11
    N_FEATURES = 4
    PITCH_L = 105.0
    PITCH_W = 68.0

    for team_label, team_id in team_map.items():
        if team_label == "referee":
            continue

        player_result = session.execute(
            sa_select(Player).where(
                Player.match_id == match_id,
                Player.team_id == team_id,
                Player.is_referee == False,
            )
        ).scalars().all()

        if not player_result:
            continue

        player_ids = [p.id for p in player_result]

        # Load all tracking points for this team, grouped by frame
        tp_result = session.execute(
            sa_select(TrackingPoint)
            .where(
                TrackingPoint.player_id.in_(player_ids),
                TrackingPoint.pitch_x != None,
            )
            .order_by(TrackingPoint.frame)
        ).scalars().all()

        if not tp_result:
            continue

        # Group by frame
        from collections import defaultdict
        frame_positions = defaultdict(list)
        for tp in tp_result:
            frame_positions[tp.frame].append((tp.pitch_x, tp.pitch_y))

        frames = sorted(frame_positions.keys())
        # Subsample to ~5fps
        frames = frames[::STEP]

        prev_formation = None

        for i, frame_no in enumerate(frames):
            # Report at intervals
            if i % max(1, int(INTERVAL_S * 5)) != 0 and i != 0:
                continue

            positions = frame_positions[frame_no]
            if len(positions) < 5:
                continue

            formation = None
            conf = 0.0
            explanation = {}

            # ── Try Transformer sliding-window inference ───────────────────────
            if use_model and i >= WINDOW:
                try:
                    import numpy as np
                    window_frames = frames[max(0, i - WINDOW):i]
                    seq = np.zeros((WINDOW, N_PLAYERS * N_FEATURES), dtype=np.float32)
                    prev_pos = None
                    for t, wf in enumerate(window_frames[-WINDOW:]):
                        pts = sorted(frame_positions[wf], key=lambda p: p[0])[:N_PLAYERS]
                        pos = np.zeros((N_PLAYERS, 2), dtype=np.float32)
                        for j, (px, py) in enumerate(pts):
                            pos[j, 0] = px / PITCH_L
                            pos[j, 1] = py / PITCH_W
                        vel = np.clip((pos - prev_pos) * 5.0, -1.0, 1.0) if prev_pos is not None else np.zeros_like(pos)
                        prev_pos = pos.copy()
                        for j in range(N_PLAYERS):
                            base = j * N_FEATURES
                            seq[t, base:base + 4] = [pos[j, 0], pos[j, 1], vel[j, 0], vel[j, 1]]

                    formation, conf, explanation = formation_model.predict(seq)
                    explanation["method"] = "transformer"
                except Exception as exc:
                    logger.debug("Transformer inference failed at frame {}: {}", frame_no, exc)
                    formation = None

            # ── Fallback: rule-based ───────────────────────────────────────────
            if not formation:
                formation, conf, explanation = detect_formation(positions)

            if formation != prev_formation:
                event = TacticalEvent(
                    match_id=match_id,
                    frame=frame_no,
                    timestamp_s=frame_no / source_fps,
                    event_type="formation_change",
                    team_label=team_label,
                    value=formation,
                    confidence=conf,
                    metadata_json=json.dumps(explanation),
                )
                session.add(event)
                prev_formation = formation

    session.commit()
