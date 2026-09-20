# Re-export SQLAlchemy ORM models.
#
# app/models/ (this package, ML architectures) shadows app/models.py (ORM).
# All code that does `from app.models import Match` etc. is satisfied here.
import importlib, sys

# Load the ORM file directly by path to avoid the package name collision
import importlib.util, pathlib
_orm_path = pathlib.Path(__file__).parent.parent / "models.py"
_spec = importlib.util.spec_from_file_location("app._orm_models", _orm_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Match = _mod.Match
Team = _mod.Team
Player = _mod.Player
TrackingPoint = _mod.TrackingPoint
BallTrackingPoint = _mod.BallTrackingPoint
TacticalEvent = _mod.TacticalEvent

__all__ = [
    "Match",
    "Team",
    "Player",
    "TrackingPoint",
    "BallTrackingPoint",
    "TacticalEvent",
]
