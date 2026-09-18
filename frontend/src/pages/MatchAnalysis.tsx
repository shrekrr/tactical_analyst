import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft, Users, Activity, Loader2, AlertCircle, RefreshCw,
} from 'lucide-react';
import {
  getMatch, getMatchAnalytics, getMatchStatus,
  getFrameSnapshot, getPlayerDetail, getVideoUrl,
} from '../services/api';
import type { MatchInfo, MatchAnalytics, FrameSnapshot, PlayerSummary, PlayerDetail } from '../types/analysis';
import { Pitch } from '../components/Pitch';
import { VideoPlayer } from '../components/VideoPlayer';
import { PlayerCard } from '../components/PlayerCard';
import { Timeline } from '../components/Timeline';
import { MetricCard, PossessionBar } from '../components/MetricCard';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';

export const MatchAnalysis: React.FC = () => {
  const { matchId } = useParams<{ matchId: string }>();
  const navigate = useNavigate();
  const [match, setMatch] = useState<MatchInfo | null>(null);
  const [analytics, setAnalytics] = useState<MatchAnalytics | null>(null);
  const [snapshot, setSnapshot] = useState<FrameSnapshot | null>(null);
  const [trailHistory, setTrailHistory] = useState<FrameSnapshot[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerSummary | null>(null);
  const [playerDetail, setPlayerDetail] = useState<PlayerDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [videoSeek, setVideoSeek] = useState<number | undefined>(undefined);
  const [activeTab, setActiveTab] = useState<'pitch' | 'heatmap' | 'shape'>('pitch');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fps = match?.fps || 25;
  const sampleFps = match?.sample_fps || 5;

  const loadMatch = useCallback(async () => {
    if (!matchId) return;
    try {
      const m = await getMatch(matchId);
      setMatch(m);
      if (m.status === 'completed') {
        const a = await getMatchAnalytics(matchId);
        setAnalytics(a);
        setLoading(false);
        if (pollRef.current) clearInterval(pollRef.current);
      } else if (m.status === 'failed') {
        setError(m.status);
        setLoading(false);
        if (pollRef.current) clearInterval(pollRef.current);
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to load match.');
      setLoading(false);
    }
  }, [matchId]);

  useEffect(() => {
    loadMatch();
    pollRef.current = setInterval(loadMatch, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [loadMatch]);

  const handleTimeUpdate = useCallback(async (currentTime: number) => {
    if (!matchId || !match) return;
    const frameNo = Math.round(currentTime * fps);
    // Snap to nearest sampled frame
    const sampledFrame = Math.round(frameNo / (fps / sampleFps)) * Math.round(fps / sampleFps);
    try {
      const snap = await getFrameSnapshot(matchId, sampledFrame);
      setSnapshot(snap);
      setTrailHistory(prev => [...prev.slice(-10), snap]);
    } catch { /* ok — frame may not exist */ }
  }, [matchId, match, fps, sampleFps]);

  const handlePlayerClick = useCallback(async (playerId: string) => {
    if (!matchId || !analytics) return;
    const player = analytics.players.find(p => p.id === playerId);
    if (!player) return;
    setSelectedPlayer(player);
    try {
      const detail = await getPlayerDetail(matchId, playerId);
      setPlayerDetail(detail);
    } catch { setPlayerDetail(null); }
  }, [matchId, analytics]);

  const handleSeek = (ts: number) => setVideoSeek(ts);

  const teamA = analytics?.teams.find(t => t.team_label === 'team_a');
  const teamB = analytics?.teams.find(t => t.team_label === 'team_b');

  const possession = (() => {
    if (!analytics) return null;
    const tl = analytics.possession_timeline;
    if (!tl.length) return null;
    let a = 0, b = 0, u = 0;
    for (const p of tl) {
      if (p.team_label === 'team_a') a++;
      else if (p.team_label === 'team_b') b++;
      else u++;
    }
    const total = a + b + u || 1;
    return { a: (a / total) * 100, b: (b / total) * 100, u: (u / total) * 100 };
  })();

  // Processing state
  if (!match || (match.status !== 'completed' && match.status !== 'failed')) {
    const prog = match?.progress || 0;
    return (
      <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center gap-6">
        <div className="text-center">
          <Loader2 className="animate-spin mx-auto mb-4 text-blue-400" size={36} />
          <h2 className="text-lg font-semibold text-slate-200">
            {match?.status === 'queued' ? 'Queued for Processing' : 'Analysing Match…'}
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            {match?.status === 'processing' ? `Stage: detection & tracking` : 'Starting soon…'}
          </p>
        </div>
        <div className="w-64 space-y-2">
          <div className="flex justify-between text-xs text-slate-500">
            <span>{match?.status || 'Loading'}</span>
            <span>{prog}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill transition-all duration-1000" style={{ width: `${prog}%` }} />
          </div>
        </div>
        <button onClick={() => navigate('/')} className="btn-secondary text-sm">
          <ArrowLeft size={14} /> Back to Dashboard
        </button>
      </div>
    );
  }

  if (error || match?.status === 'failed') {
    return (
      <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center gap-4">
        <AlertCircle className="text-red-400" size={36} />
        <h2 className="text-lg font-semibold text-slate-200">Processing Failed</h2>
        <p className="text-sm text-slate-500 max-w-sm text-center">
          {error || 'The analysis pipeline encountered an error. Please try again with a different video.'}
        </p>
        <button onClick={() => navigate('/')} className="btn-secondary text-sm">
          <ArrowLeft size={14} /> Back
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-900 flex">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 border-r border-slate-700/50 flex flex-col p-4 gap-1">
        <button onClick={() => navigate('/')} className="nav-item mb-4">
          <ArrowLeft size={15} /> All Matches
        </button>
        <div className="px-3 py-2">
          <div className="text-xs font-medium text-slate-300 truncate">{match.filename}</div>
          <div className="text-xs text-slate-500 mt-0.5">
            {match.duration_s ? `${Math.floor(match.duration_s / 60)}min` : '—'} · {match.fps?.toFixed(0)}fps
          </div>
        </div>
        <div className="mt-2 border-t border-slate-700/50 pt-2 space-y-0.5">
          <div className="nav-item active"><Activity size={15} /> Analysis</div>
          <div className="nav-item" onClick={() => navigate(`/players/${matchId}`)}>
            <Users size={15} /> Players
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-5">
        <div className="max-w-7xl mx-auto space-y-5">
          {/* Top row: Video + Pitch */}
          <div className="grid grid-cols-2 gap-5">
            {/* Video */}
            <div className="space-y-3">
              <VideoPlayer
                src={getVideoUrl(match.filename)}
                onTimeUpdate={(t) => handleTimeUpdate(t)}
                seekTo={videoSeek}
              />
            </div>

            {/* Pitch */}
            <div className="space-y-3">
              <div className="card-header px-4 py-2">
                <span className="text-xs font-medium text-slate-300 uppercase tracking-widest">2D Pitch</span>
                <div className="flex gap-2">
                  {[['pitch', 'Positions'], ['heatmap', 'Heatmap']].map(([k, l]) => (
                    <button
                      key={k}
                      onClick={() => setActiveTab(k as any)}
                      className={`text-xs px-2 py-0.5 rounded transition-colors ${activeTab === k ? 'bg-blue-600/30 text-blue-300' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>
              <div className="relative">
                <Pitch
                  snapshot={snapshot}
                  showTrails={activeTab === 'pitch'}
                  trailHistory={trailHistory}
                  selectedPlayerId={selectedPlayer?.id}
                  onPlayerClick={handlePlayerClick}
                  className="w-full aspect-[105/68]"
                />
              </div>
            </div>
          </div>

          {/* Match stats row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="Avg Team Width"
              value={teamA?.avg_width_m?.toFixed(1) ?? null}
              unit="m"
              color="#3b82f6"
            />
            <MetricCard
              label="Avg Team Depth"
              value={teamA?.avg_depth_m?.toFixed(1) ?? null}
              unit="m"
              color="#06b6d4"
            />
            <MetricCard
              label="Defensive Line"
              value={teamA?.avg_defensive_line_m?.toFixed(1) ?? null}
              unit="m"
              color="#f59e0b"
            />
            <MetricCard
              label="Compactness"
              value={teamA?.avg_compactness_m?.toFixed(1) ?? null}
              unit="m avg"
              color="#8b5cf6"
            />
          </div>

          {/* Possession + Formations row */}
          <div className="grid grid-cols-3 gap-5">
            <div className="col-span-2 card p-4 space-y-4">
              {possession && teamA && teamB && (
                <PossessionBar
                  teamA={{ label: teamA.display_name || 'Team A', color: teamA.color_hex, pct: possession.a }}
                  teamB={{ label: teamB.display_name || 'Team B', color: teamB.color_hex, pct: possession.b }}
                  unknown={possession.u}
                />
              )}

              {/* Formation badges */}
              <div className="flex gap-4">
                {[teamA, teamB].filter(Boolean).map(team => (
                  <div key={team!.team_label} className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full" style={{ background: team!.color_hex }} />
                    <span className="text-xs text-slate-400">{team!.display_name}</span>
                    {team!.current_formation && (
                      <span className="badge-formation">{team!.current_formation}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-4">
              <div className="stat-label mb-3">Formation Timeline</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {analytics?.formation_timeline.slice(0, 10).map((ev, i) => {
                  const t = Math.floor(ev.timestamp_s / 60);
                  const s = Math.floor(ev.timestamp_s % 60);
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-500 font-mono w-10">{t}:{s.toString().padStart(2, '0')}</span>
                      <span className="badge-formation">{ev.formation}</span>
                      <span className="text-slate-500 capitalize text-xs">{ev.team_label?.replace('_', ' ')}</span>
                    </div>
                  );
                })}
                {(!analytics?.formation_timeline.length) && (
                  <span className="text-xs text-slate-600 italic">No changes detected</span>
                )}
              </div>
            </div>
          </div>

          {/* Timeline */}
          <Timeline
            events={analytics?.tactical_events || []}
            duration_s={match.duration_s || 0}
            onSeek={handleSeek}
          />

          {/* Player table */}
          <div className="card overflow-hidden">
            <div className="card-header">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-widest">Players</span>
              <span className="text-xs text-slate-500">{analytics?.players.filter(p => !p.is_referee).length} detected</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    {['Team', 'ID', 'Distance', 'Avg Speed', 'Top Speed', 'DEF%', 'MID%', 'ATK%'].map(h => (
                      <th key={h} className="px-4 py-2 text-left text-slate-500 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {analytics?.players.filter(p => !p.is_referee).map(player => (
                    <tr
                      key={player.id}
                      className={`border-b border-slate-700/30 hover:bg-surface-700/50 cursor-pointer transition-colors ${selectedPlayer?.id === player.id ? 'bg-blue-500/10' : ''}`}
                      onClick={() => handlePlayerClick(player.id)}
                    >
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ background: player.team_color || '#6b7280' }} />
                          <span className="capitalize text-slate-400">{player.team_label?.replace('_', ' ')}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2 font-mono text-slate-300">#{player.tracking_id}</td>
                      <td className="px-4 py-2 text-slate-300">
                        {player.total_distance_m ? `${(player.total_distance_m / 1000).toFixed(2)} km` : '—'}
                      </td>
                      <td className="px-4 py-2 text-slate-300">
                        {player.avg_speed_kmh ? `${player.avg_speed_kmh.toFixed(1)} km/h` : '—'}
                      </td>
                      <td className="px-4 py-2 text-slate-300">
                        {player.max_speed_kmh ? `${player.max_speed_kmh.toFixed(1)} km/h` : '—'}
                      </td>
                      <td className="px-4 py-2 text-red-400">{player.pct_defensive_third?.toFixed(0) ?? '—'}%</td>
                      <td className="px-4 py-2 text-yellow-400">{player.pct_middle_third?.toFixed(0) ?? '—'}%</td>
                      <td className="px-4 py-2 text-emerald-400">{player.pct_attacking_third?.toFixed(0) ?? '—'}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>

      {/* Player detail panel */}
      {selectedPlayer && (
        <aside className="w-72 shrink-0 border-l border-slate-700/50 overflow-auto">
          <PlayerCard
            player={selectedPlayer}
            detail={playerDetail}
            onClose={() => { setSelectedPlayer(null); setPlayerDetail(null); }}
          />
        </aside>
      )}
    </div>
  );
};
