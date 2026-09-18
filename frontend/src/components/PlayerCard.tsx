import React from 'react';
import type { PlayerSummary, PlayerDetail } from '../types/analysis';
import { X, User, Zap, MapPin, Activity } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface PlayerCardProps {
  player: PlayerSummary;
  detail?: PlayerDetail | null;
  onClose: () => void;
}

const ZoneBar: React.FC<{ label: string; pct: number | null; color: string }> = ({
  label, pct, color,
}) => (
  <div className="flex items-center gap-2 text-xs">
    <span className="text-slate-400 w-24 shrink-0">{label}</span>
    <div className="flex-1 progress-bar">
      <div className="progress-fill" style={{ width: `${pct ?? 0}%`, background: color }} />
    </div>
    <span className="text-slate-300 w-10 text-right">{pct?.toFixed(0) ?? '—'}%</span>
  </div>
);

export const PlayerCard: React.FC<PlayerCardProps> = ({ player, detail, onClose }) => {
  const teamColor = player.team_color || '#6b7280';
  const speedData = detail?.speed_series.slice(-100).map(s => ({
    t: s.timestamp_s.toFixed(0),
    speed: s.speed_kmh?.toFixed(1) ?? 0,
  })) ?? [];

  return (
    <div className="card flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="card-header">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white"
            style={{ background: teamColor }}
          >
            {player.tracking_id}
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">
              {player.is_referee ? 'Referee' : player.is_goalkeeper ? 'Goalkeeper' : `Player #${player.tracking_id}`}
            </div>
            <div className="text-xs text-slate-400 capitalize">
              {player.team_label?.replace('_', ' ') ?? 'Unknown team'}
            </div>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
          <X size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Key stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-surface-700 rounded-lg p-3">
            <div className="stat-label flex items-center gap-1"><Activity size={10} /> Distance</div>
            <div className="stat-value mt-1">
              {player.total_distance_m != null
                ? (player.total_distance_m / 1000).toFixed(2)
                : '—'}
              <span className="stat-unit">km</span>
            </div>
          </div>
          <div className="bg-surface-700 rounded-lg p-3">
            <div className="stat-label flex items-center gap-1"><Zap size={10} /> Top Speed</div>
            <div className="stat-value mt-1">
              {player.max_speed_kmh?.toFixed(1) ?? '—'}
              <span className="stat-unit">km/h</span>
            </div>
          </div>
          <div className="bg-surface-700 rounded-lg p-3">
            <div className="stat-label flex items-center gap-1"><Zap size={10} /> Avg Speed</div>
            <div className="stat-value mt-1">
              {player.avg_speed_kmh?.toFixed(1) ?? '—'}
              <span className="stat-unit">km/h</span>
            </div>
          </div>
          <div className="bg-surface-700 rounded-lg p-3">
            <div className="stat-label flex items-center gap-1"><MapPin size={10} /> Avg Position</div>
            <div className="text-sm font-mono text-slate-200 mt-1">
              x: {player.avg_x?.toFixed(1) ?? '—'}m<br />
              y: {player.avg_y?.toFixed(1) ?? '—'}m
            </div>
          </div>
        </div>

        {/* Zone breakdown */}
        <div>
          <div className="stat-label mb-3">Zone Distribution</div>
          <div className="space-y-2">
            <ZoneBar label="Defensive Third" pct={player.pct_defensive_third} color="#ef4444" />
            <ZoneBar label="Middle Third" pct={player.pct_middle_third} color="#f59e0b" />
            <ZoneBar label="Attacking Third" pct={player.pct_attacking_third} color="#22c55e" />
          </div>
        </div>

        {/* Speed chart */}
        {speedData.length > 3 && (
          <div>
            <div className="stat-label mb-3">Speed Profile (est.)</div>
            <div className="h-24">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={speedData} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                  <defs>
                    <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={teamColor} stopOpacity={0.5} />
                      <stop offset="95%" stopColor={teamColor} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, 'dataMax + 5']} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #334155', borderRadius: 6 }}
                    labelStyle={{ color: '#94a3b8' }}
                    itemStyle={{ color: teamColor }}
                    formatter={(v: number) => [`${v} km/h`, 'Speed']}
                  />
                  <Area
                    type="monotone"
                    dataKey="speed"
                    stroke={teamColor}
                    strokeWidth={1.5}
                    fill="url(#speedGrad)"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-slate-500 mt-1 italic">AI-estimated speed — not GPS-grade accuracy</p>
          </div>
        )}
      </div>
    </div>
  );
};
