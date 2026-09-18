"""
WebSocket endpoint for real-time processing progress.
"""
import asyncio
import json
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

# In-memory registry: match_id → set of connected WebSocket clients
_connections: Dict[str, set[WebSocket]] = {}


@router.websocket("/matches/{match_id}/progress")
async def progress_ws(websocket: WebSocket, match_id: str):
    """WebSocket endpoint; clients receive progress updates for a match."""
    await websocket.accept()
    _connections.setdefault(match_id, set()).add(websocket)
    logger.debug("WS connected for match {}", match_id)
    try:
        while True:
            # Keep connection alive; data is pushed via broadcast()
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        _connections[match_id].discard(websocket)
        logger.debug("WS disconnected for match {}", match_id)


async def broadcast(match_id: str, payload: dict) -> None:
    """Broadcast a progress message to all connected clients for a match."""
    sockets = _connections.get(match_id, set())
    dead = set()
    for ws in sockets:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.add(ws)
    sockets -= dead
