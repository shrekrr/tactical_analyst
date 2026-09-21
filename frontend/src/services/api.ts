import axios from 'axios';
import type {
  CalibrationPoint,
  CalibrationResponse,
  FrameSnapshot,
  HealthResponse,
  MatchAnalytics,
  MatchInfo,
  MatchStatusResponse,
  MatchUploadResponse,
  PlayerDetail,
  PlayerSummary,
} from '../types/analysis';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// ── Health ────────────────────────────────────────────────────────────────────

export const getHealth = (): Promise<HealthResponse> =>
  api.get('/api/health').then(r => r.data);

// ── Matches ───────────────────────────────────────────────────────────────────

export const listMatches = (): Promise<MatchInfo[]> =>
  api.get('/api/matches/').then(r => r.data);

export const getMatch = (matchId: string): Promise<MatchInfo> =>
  api.get(`/api/matches/${matchId}`).then(r => r.data);

export const getMatchStatus = (matchId: string): Promise<MatchStatusResponse> =>
  api.get(`/api/matches/${matchId}/status`).then(r => r.data);

export const getMatchAnalytics = (matchId: string): Promise<MatchAnalytics> =>
  api.get(`/api/matches/${matchId}/analytics`).then(r => r.data);

export const getFrameSnapshot = (matchId: string, frame: number): Promise<FrameSnapshot> =>
  api.get(`/api/matches/${matchId}/frame/${frame}`).then(r => r.data);

// ── Upload ────────────────────────────────────────────────────────────────────

export const uploadVideo = (
  file: File,
  onProgress?: (pct: number) => void,
): Promise<MatchUploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/matches/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  }).then(r => r.data);
};

// ── Analysis ──────────────────────────────────────────────────────────────────

export const startAnalysis = (
  matchId: string,
  sampleFps: number = 5,
): Promise<MatchStatusResponse> =>
  api.post(`/api/matches/${matchId}/analyze`, { sample_fps: sampleFps }).then(r => r.data);

// ── Calibration ───────────────────────────────────────────────────────────────

export const submitCalibration = (
  matchId: string,
  points: CalibrationPoint[],
): Promise<CalibrationResponse> =>
  api.post(`/api/matches/${matchId}/calibrate`, { points }).then(r => r.data);

// ── Players ───────────────────────────────────────────────────────────────────

export const listPlayers = (matchId: string): Promise<PlayerSummary[]> =>
  api.get(`/api/matches/${matchId}/players`).then(r => r.data);

export const getPlayerDetail = (matchId: string, playerId: string): Promise<PlayerDetail> =>
  api.get(`/api/matches/${matchId}/players/${playerId}`).then(r => r.data);

// ── Video URL helper ──────────────────────────────────────────────────────────

export const getVideoUrl = (idOrFilename: string): string => {
  if (!idOrFilename) return '';
  const clean = idOrFilename.replace(/^\/videos\//, '');
  if (clean.includes('.')) {
    return `${BASE_URL}/videos/${clean}`;
  }
  return `${BASE_URL}/videos/${clean}.mp4`;
};



// ── WebSocket helper ──────────────────────────────────────────────────────────

export const createProgressSocket = (
  matchId: string,
  onMessage: (data: { type: string; progress?: number; stage?: string; message?: string }) => void,
): WebSocket => {
  const wsBase = BASE_URL.replace('http', 'ws');
  const ws = new WebSocket(`${wsBase}/ws/matches/${matchId}/progress`);
  ws.onmessage = e => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {
      // ignore malformed messages
    }
  };
  return ws;
};
