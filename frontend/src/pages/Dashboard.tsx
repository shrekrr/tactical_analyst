import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Play, Clock, CheckCircle, Loader2, AlertCircle, Cpu } from 'lucide-react';
import { listMatches, getHealth, startAnalysis } from '../services/api';
import type { MatchInfo, HealthResponse } from '../types/analysis';

function formatBytes(b: number | null): string {
  if (b == null) return '—';
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  if (b < 1e9) return `${(b / 1e6).toFixed(0)} MB`;
  return `${(b / 1e9).toFixed(2)} GB`;
}

function formatDuration(s: number | null): string {
  if (s == null) return '—';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

const StatusBadge: React.FC<{ status: MatchInfo['status']; progress: number }> = ({ status, progress }) => {
  const map = {
    uploaded:   { icon: <Clock size={11} />,   color: 'text-slate-400 bg-slate-700',        label: 'Uploaded' },
    queued:     { icon: <Clock size={11} />,     color: 'text-yellow-400 bg-yellow-400/10',  label: 'Queued' },
    processing: { icon: <Loader2 size={11} className="animate-spin" />, color: 'text-blue-400 bg-blue-400/10', label: `Processing ${progress}%` },
    completed:  { icon: <CheckCircle size={11} />, color: 'text-emerald-400 bg-emerald-400/10', label: 'Completed' },
    failed:     { icon: <AlertCircle size={11} />, color: 'text-red-400 bg-red-400/10',       label: 'Failed' },
  };
  const cfg = map[status] || map.uploaded;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${cfg.color}`}>
      {cfg.icon}{cfg.label}
    </span>
  );
};

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [matches, setMatches] = useState<MatchInfo[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const loadMatches = React.useCallback(async () => {
    try {
      const [m, h] = await Promise.all([listMatches(), getHealth()]);
      setMatches(m);
      setHealth(h);
    } catch {
      // Server may not be running yet
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMatches();
    // Poll for status updates every 5s
    const interval = setInterval(loadMatches, 5000);
    return () => clearInterval(interval);
  }, [loadMatches]);

  return (
    <div className="min-h-screen bg-surface-900 flex">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 border-r border-slate-700/50 flex flex-col p-4 gap-1">
        <div className="flex items-center gap-2.5 mb-6 px-1">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">E</div>
          <div>
            <div className="text-sm font-semibold text-slate-100 leading-none">EPL Analyst</div>
            <div className="text-xs text-slate-500 leading-none mt-0.5">Tactical AI</div>
          </div>
        </div>

        <nav className="flex-1 space-y-0.5">
          <div className="nav-item active">
            <Play size={15} /> Dashboard
          </div>
          <div className="nav-item" onClick={() => navigate('/upload')}>
            <Plus size={15} /> Upload Match
          </div>
        </nav>

        {/* Device info */}
        {health && (
          <div className="mt-auto pt-4 border-t border-slate-700/50">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Cpu size={12} />
              <span>{health.gpu || health.compute_device}</span>
            </div>
            {health.gpu && (
              <div className="text-xs text-emerald-400/70 mt-0.5 pl-5">CUDA enabled</div>
            )}
          </div>
        )}
      </aside>

      {/* Main */}
      <main className="flex-1 p-6 overflow-auto">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Matches</h2>
              <p className="text-sm text-slate-500">
                {matches.length} match{matches.length !== 1 ? 'es' : ''} total
              </p>
            </div>
            <button onClick={() => navigate('/upload')} className="btn-primary">
              <Plus size={15} /> Upload Match
            </button>
          </div>

          {/* Match list */}
          {loading ? (
            <div className="flex items-center justify-center py-20 text-slate-500">
              <Loader2 className="animate-spin mr-2" size={18} />
              Loading…
            </div>
          ) : matches.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-2xl bg-surface-700 flex items-center justify-center mb-4">
                <Play size={28} className="text-slate-600" />
              </div>
              <h3 className="text-base font-medium text-slate-400 mb-2">No matches yet</h3>
              <p className="text-sm text-slate-600 mb-6 max-w-xs">
                Upload a football match video to get started with AI tactical analysis.
              </p>
              <button onClick={() => navigate('/upload')} className="btn-primary">
                <Plus size={15} /> Upload your first match
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {matches.map(match => (
                  <div
                    key={match.id}
                    className="card p-4 flex items-center gap-4 cursor-pointer hover:border-slate-600 transition-colors"
                    onClick={() => navigate(`/match/${match.id}`)}
                  >
                    {/* Video thumbnail placeholder */}
                    <div className="w-20 h-14 bg-surface-700 rounded-lg flex items-center justify-center shrink-0">
                      <Play size={18} className="text-slate-500" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-200 truncate">{match.filename}</div>
                      <div className="flex items-center gap-3 mt-1.5">
                        <span className="text-xs text-slate-500">{formatDuration(match.duration_s)}</span>
                        <span className="text-xs text-slate-600">•</span>
                        <span className="text-xs text-slate-500">{formatBytes(match.file_size_bytes)}</span>
                        {match.fps && (
                          <>
                            <span className="text-xs text-slate-600">•</span>
                            <span className="text-xs text-slate-500">{match.fps?.toFixed(0)} FPS</span>
                          </>
                        )}
                        {match.width && (
                          <>
                            <span className="text-xs text-slate-600">•</span>
                            <span className="text-xs text-slate-500">{match.width}×{match.height}</span>
                          </>
                        )}
                      </div>

                      {/* Progress bar for processing */}
                      {match.status === 'processing' && (
                        <div className="mt-2 progress-bar w-48">
                          <div className="progress-fill" style={{ width: `${match.progress}%` }} />
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-3">
                      <StatusBadge status={match.status} progress={match.progress} />
                      {match.status === 'completed' && (
                        <button
                          className="btn-secondary text-xs"
                          onClick={e => { e.stopPropagation(); navigate(`/match/${match.id}`); }}
                        >
                          View Analysis →
                        </button>
                      )}
                      {(match.status === 'queued' || match.status === 'processing') && (
                        <button
                          className="btn-secondary text-xs flex items-center gap-1.5"
                          onClick={e => { e.stopPropagation(); navigate(`/match/${match.id}`); }}
                        >
                          <Loader2 size={12} className="animate-spin text-accent-green" />
                          View Progress →
                        </button>
                      )}
                      {match.status === 'failed' && (
                        <button
                          className="btn-primary text-xs"
                          onClick={async e => {
                            e.stopPropagation();
                            try {
                              await startAnalysis(match.id);
                              loadMatches();
                            } catch (err) {
                              console.error(err);
                            }
                          }}
                        >
                          Retry Analysis
                        </button>
                      )}
                      {match.status === 'uploaded' && (
                        <button
                          className="btn-primary text-xs"
                          onClick={async e => {
                            e.stopPropagation();
                            try {
                              await startAnalysis(match.id);
                              loadMatches();
                            } catch (err) {
                              console.error(err);
                            }
                          }}
                        >
                          Start Analysis
                        </button>
                      )}
                    </div>
                  </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
