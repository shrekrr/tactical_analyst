import React from 'react';
import type { TacticalEvent } from '../types/analysis';

interface TimelineProps {
  events: TacticalEvent[];
  duration_s: number;
  currentTime?: number;
  onSeek?: (timestamp_s: number) => void;
}

const EVENT_COLORS: Record<string, string> = {
  formation_change: '#3b82f6',
  tactical_phase: '#f59e0b',
  possession_change: '#22c55e',
};

const EVENT_ICONS: Record<string, string> = {
  formation_change: '⬡',
  tactical_phase: '⚡',
  possession_change: '●',
};

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}

export const Timeline: React.FC<TimelineProps> = ({
  events,
  duration_s,
  currentTime = 0,
  onSeek,
}) => {
  const dur = duration_s || 1;

  // Minute markers
  const minutes = Math.ceil(dur / 60);
  const markerStep = minutes <= 45 ? 5 : minutes <= 90 ? 15 : 30;
  const markers: number[] = [];
  for (let m = 0; m <= minutes; m += markerStep) {
    markers.push(m * 60);
  }

  return (
    <div className="card p-4">
      <div className="stat-label mb-4">Match Timeline</div>

      {/* Timeline bar */}
      <div className="relative">
        {/* Background track */}
        <div className="h-2 bg-surface-600 rounded-full relative overflow-visible">
          {/* Progress */}
          <div
            className="absolute inset-y-0 left-0 bg-blue-500/50 rounded-full transition-all"
            style={{ width: `${(currentTime / dur) * 100}%` }}
          />

          {/* Current time indicator */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-blue-400 rounded-full border-2 border-slate-900 shadow transition-all"
            style={{ left: `${(currentTime / dur) * 100}%`, transform: 'translateX(-50%) translateY(-50%)' }}
          />
        </div>

        {/* Event dots */}
        {events.map(event => {
          const pct = (event.timestamp_s / dur) * 100;
          const color = EVENT_COLORS[event.event_type] || '#6b7280';
          return (
            <button
              key={event.id}
              onClick={() => onSeek?.(event.timestamp_s)}
              className="absolute top-0 -translate-y-1 group"
              style={{ left: `${pct}%`, transform: `translateX(-50%) translateY(-25%)` }}
              title={`${event.event_type}: ${event.value} @ ${formatTime(event.timestamp_s)}`}
            >
              <div
                className="w-3 h-3 rounded-full border-2 border-slate-900 transition-transform group-hover:scale-150"
                style={{ background: color }}
              />
              {/* Tooltip */}
              <div className="absolute bottom-5 left-1/2 -translate-x-1/2 bg-surface-700 border border-slate-600 rounded px-2 py-1 text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                <div className="font-medium" style={{ color }}>{event.event_type.replace('_', ' ')}</div>
                <div className="text-slate-300">{event.value}</div>
                <div className="text-slate-500">{formatTime(event.timestamp_s)}</div>
              </div>
            </button>
          );
        })}

        {/* Minute markers */}
        <div className="relative mt-2">
          {markers.map(ts => (
            <button
              key={ts}
              onClick={() => onSeek?.(ts)}
              className="absolute text-xs text-slate-500 hover:text-slate-300 transition-colors -translate-x-1/2"
              style={{ left: `${(ts / dur) * 100}%` }}
            >
              {formatTime(ts)}
            </button>
          ))}
        </div>
      </div>

      {/* Event list */}
      {events.length > 0 && (
        <div className="mt-8 space-y-1 max-h-36 overflow-y-auto">
          {events.map(event => {
            const color = EVENT_COLORS[event.event_type] || '#6b7280';
            return (
              <button
                key={event.id}
                onClick={() => onSeek?.(event.timestamp_s)}
                className="w-full flex items-center gap-3 px-2 py-1.5 rounded hover:bg-surface-600 transition-colors text-left"
              >
                <span className="text-xs font-mono text-slate-500 w-12 shrink-0">
                  {formatTime(event.timestamp_s)}
                </span>
                <span className="text-xs" style={{ color }}>
                  {EVENT_ICONS[event.event_type] || '•'}
                </span>
                <span className="text-xs text-slate-400 capitalize">
                  {event.event_type.replace(/_/g, ' ')}
                </span>
                {event.value && (
                  <span className="text-xs font-medium text-slate-200">{event.value}</span>
                )}
                {event.team_label && (
                  <span className="text-xs text-slate-500 capitalize ml-auto">
                    {event.team_label.replace('_', ' ')}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {events.length === 0 && (
        <div className="mt-6 text-center text-sm text-slate-600 italic">
          No events detected yet
        </div>
      )}
    </div>
  );
};
