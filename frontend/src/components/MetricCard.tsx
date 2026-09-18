import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number | null;
  unit?: string;
  subtext?: string;
  icon?: React.ReactNode;
  color?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  subtext,
  icon,
  color = '#3b82f6',
  trend,
  className = '',
}) => {
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : null;
  const trendColor = trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-slate-400';

  return (
    <div className={`bg-surface-800 border border-slate-700/50 rounded-lg p-4 ${className}`}>
      <div className="flex items-start justify-between">
        <div className="stat-label">{label}</div>
        {icon && (
          <div className="text-slate-500 opacity-70">{icon}</div>
        )}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        {value != null ? (
          <>
            <span
              className="text-xl font-bold leading-none"
              style={{ color }}
            >
              {typeof value === 'number' ? value.toFixed(value < 10 ? 1 : 0) : value}
            </span>
            {unit && <span className="text-sm text-slate-400">{unit}</span>}
            {trendIcon && <span className={`text-sm ml-1 ${trendColor}`}>{trendIcon}</span>}
          </>
        ) : (
          <span className="text-xl font-bold text-slate-600">—</span>
        )}
      </div>
      {subtext && (
        <div className="text-xs text-slate-500 mt-1">{subtext}</div>
      )}
    </div>
  );
};

interface PossessionBarProps {
  teamA: { label: string; color: string; pct: number };
  teamB: { label: string; color: string; pct: number };
  unknown?: number;
}

export const PossessionBar: React.FC<PossessionBarProps> = ({ teamA, teamB, unknown = 0 }) => {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-1.5">
        <span className="font-medium" style={{ color: teamA.color }}>{teamA.label}</span>
        <span className="text-slate-500">AI-estimated possession</span>
        <span className="font-medium" style={{ color: teamB.color }}>{teamB.label}</span>
      </div>
      <div className="h-3 rounded-full overflow-hidden flex">
        <div
          className="h-full transition-all duration-500"
          style={{ width: `${teamA.pct}%`, background: teamA.color }}
        />
        {unknown > 0 && (
          <div className="h-full bg-slate-600" style={{ width: `${unknown}%` }} />
        )}
        <div
          className="h-full transition-all duration-500"
          style={{ width: `${teamB.pct}%`, background: teamB.color }}
        />
      </div>
      <div className="flex justify-between text-xs mt-1">
        <span className="font-bold" style={{ color: teamA.color }}>{teamA.pct.toFixed(0)}%</span>
        <span className="font-bold" style={{ color: teamB.color }}>{teamB.pct.toFixed(0)}%</span>
      </div>
    </div>
  );
};
