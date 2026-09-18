import React, { useEffect, useRef } from 'react';
import type { FrameSnapshot } from '../types/analysis';

interface PitchProps {
  snapshot: FrameSnapshot | null;
  showTrails?: boolean;
  trailHistory?: FrameSnapshot[];
  selectedPlayerId?: string | null;
  onPlayerClick?: (playerId: string) => void;
  className?: string;
}

const PITCH_L = 105;
const PITCH_W = 68;

function toSvg(pitchX: number, pitchY: number, svgW: number, svgH: number) {
  return {
    x: (pitchX / PITCH_L) * svgW,
    y: (pitchY / PITCH_W) * svgH,
  };
}

const SVG_W = 700;
const SVG_H = 454;

export const Pitch: React.FC<PitchProps> = ({
  snapshot,
  showTrails = true,
  trailHistory = [],
  selectedPlayerId,
  onPlayerClick,
  className = '',
}) => {
  // Build trail data per player
  const trails: Record<string, { x: number; y: number; color: string }[]> = {};
  if (showTrails) {
    for (const snap of trailHistory) {
      for (const p of snap.players) {
        if (p.pitch_x == null || p.pitch_y == null) continue;
        if (!trails[p.player_id]) trails[p.player_id] = [];
        const pt = toSvg(p.pitch_x, p.pitch_y, SVG_W, SVG_H);
        trails[p.player_id].push({ ...pt, color: p.team_color });
      }
    }
  }

  return (
    <div className={`relative pitch-container rounded-lg overflow-hidden ${className}`}>
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full h-full"
        style={{ display: 'block' }}
      >
        {/* Pitch background */}
        <defs>
          <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0d3a1f" />
            <stop offset="50%" stopColor="#0f4422" />
            <stop offset="100%" stopColor="#0d3a1f" />
          </linearGradient>
          {/* Stripe pattern */}
          <pattern id="stripes" x="0" y="0" width="53.8" height="454" patternUnits="userSpaceOnUse">
            <rect x="0" y="0" width="53.8" height="454" fill="#0f4422" />
            <rect x="26.9" y="0" width="26.9" height="454" fill="#0d3a1f" />
          </pattern>
        </defs>

        <rect width={SVG_W} height={SVG_H} fill="url(#stripes)" />

        {/* Pitch markings */}
        <g stroke="#ffffff" strokeWidth="1.5" fill="none" opacity="0.5">
          {/* Outer boundary */}
          <rect x="10" y="10" width={SVG_W - 20} height={SVG_H - 20} />

          {/* Centre line */}
          <line x1={SVG_W / 2} y1="10" x2={SVG_W / 2} y2={SVG_H - 10} />

          {/* Centre circle */}
          <circle cx={SVG_W / 2} cy={SVG_H / 2} r="58" />
          <circle cx={SVG_W / 2} cy={SVG_H / 2} r="3" fill="white" stroke="none" />

          {/* Left penalty area */}
          <rect x="10" y={SVG_H / 2 - 103} width="100" height="206" />
          {/* Left 6-yard box */}
          <rect x="10" y={SVG_H / 2 - 44} width="35" height="88" />
          {/* Left penalty spot */}
          <circle cx="85" cy={SVG_H / 2} r="3" fill="white" stroke="none" />

          {/* Right penalty area */}
          <rect x={SVG_W - 110} y={SVG_H / 2 - 103} width="100" height="206" />
          {/* Right 6-yard box */}
          <rect x={SVG_W - 45} y={SVG_H / 2 - 44} width="35" height="88" />
          {/* Right penalty spot */}
          <circle cx={SVG_W - 85} cy={SVG_H / 2} r="3" fill="white" stroke="none" />

          {/* Corner arcs */}
          <path d="M10,18 Q18,18 18,26" />
          <path d={`M10,${SVG_H - 18} Q18,${SVG_H - 18} 18,${SVG_H - 26}`} />
          <path d={`M${SVG_W - 10},18 Q${SVG_W - 18},18 ${SVG_W - 18},26`} />
          <path d={`M${SVG_W - 10},${SVG_H - 18} Q${SVG_W - 18},${SVG_H - 18} ${SVG_W - 18},${SVG_H - 26}`} />

          {/* Goals */}
          <rect x="1" y={SVG_H / 2 - 30} width="9" height="60" opacity="0.7" />
          <rect x={SVG_W - 10} y={SVG_H / 2 - 30} width="9" height="60" opacity="0.7" />
        </g>

        {/* Player trails */}
        {Object.entries(trails).map(([pid, points]) => {
          if (points.length < 2) return null;
          const d = points
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
            .join(' ');
          return (
            <path
              key={`trail-${pid}`}
              d={d}
              stroke={points[0].color}
              strokeWidth="1.5"
              fill="none"
              opacity="0.4"
              strokeDasharray="none"
            />
          );
        })}

        {/* Ball */}
        {snapshot?.ball && snapshot.ball.pitch_x != null && (
          <g>
            <circle
              cx={toSvg(snapshot.ball.pitch_x, snapshot.ball.pitch_y, SVG_W, SVG_H).x}
              cy={toSvg(snapshot.ball.pitch_x, snapshot.ball.pitch_y, SVG_W, SVG_H).y}
              r="7"
              fill="white"
              stroke="#e2e8f0"
              strokeWidth="1"
            />
            <circle
              cx={toSvg(snapshot.ball.pitch_x, snapshot.ball.pitch_y, SVG_W, SVG_H).x}
              cy={toSvg(snapshot.ball.pitch_x, snapshot.ball.pitch_y, SVG_W, SVG_H).y}
              r="9"
              fill="none"
              stroke="rgba(255,255,255,0.3)"
              strokeWidth="2"
            />
          </g>
        )}

        {/* Players */}
        {snapshot?.players.map((player) => {
          if (player.pitch_x == null || player.pitch_y == null) return null;
          const pos = toSvg(player.pitch_x, player.pitch_y, SVG_W, SVG_H);
          const isSelected = selectedPlayerId === player.player_id;
          const color = player.is_referee ? '#facc15' : (player.team_color || '#6b7280');
          const r = player.is_referee ? 6 : player.is_goalkeeper ? 10 : 8;

          return (
            <g
              key={player.player_id}
              onClick={() => onPlayerClick?.(player.player_id)}
              style={{ cursor: onPlayerClick ? 'pointer' : 'default' }}
            >
              {isSelected && (
                <circle cx={pos.x} cy={pos.y} r={r + 5} fill="none" stroke="white" strokeWidth="2" opacity="0.8" />
              )}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r}
                fill={color}
                stroke="rgba(0,0,0,0.6)"
                strokeWidth="1.5"
              />
              {player.is_goalkeeper && (
                <circle cx={pos.x} cy={pos.y} r={r - 3} fill="none" stroke="rgba(0,0,0,0.5)" strokeWidth="1" />
              )}
              <text
                x={pos.x}
                y={pos.y + 3}
                textAnchor="middle"
                fill="white"
                fontSize="7"
                fontWeight="bold"
                style={{ pointerEvents: 'none' }}
              >
                {player.tracking_id}
              </text>
            </g>
          );
        })}

        {/* Zone labels */}
        <g opacity="0.25" fontSize="11" fill="white" textAnchor="middle">
          <text x={SVG_W * 0.17} y={SVG_H - 18}>DEF</text>
          <text x={SVG_W * 0.5} y={SVG_H - 18}>MID</text>
          <text x={SVG_W * 0.83} y={SVG_H - 18}>ATK</text>
          <line x1={SVG_W * 0.333} y1="10" x2={SVG_W * 0.333} y2={SVG_H - 10} stroke="white" strokeWidth="0.5" strokeDasharray="4,6" />
          <line x1={SVG_W * 0.667} y1="10" x2={SVG_W * 0.667} y2={SVG_H - 10} stroke="white" strokeWidth="0.5" strokeDasharray="4,6" />
        </g>
      </svg>

      {/* Formation overlay */}
      {(snapshot?.formation_a || snapshot?.formation_b) && (
        <div className="absolute top-2 left-2 right-2 flex justify-between px-2">
          {snapshot.formation_a && (
            <span className="badge-formation text-xs">{snapshot.formation_a}</span>
          )}
          {snapshot.formation_b && (
            <span className="badge bg-red-500/20 text-red-300 border border-red-500/30 text-xs">
              {snapshot.formation_b}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
