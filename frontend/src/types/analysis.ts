// TypeScript types for the EPL Tactical Analyst

export interface MatchUploadResponse {
  match_id: string;
  status: string;
  filename: string;
  file_size_bytes: number;
}

export interface MatchInfo {
  id: string;
  filename: string;
  duration_s: number | null;
  fps: number | null;
  total_frames: number | null;
  width: number | null;
  height: number | null;
  file_size_bytes: number | null;
  status: 'uploaded' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  sample_fps: number;
  created_at: string;
}

export interface MatchStatusResponse {
  match_id: string;
  status: string;
  progress: number;
  error_message: string | null;
}

export interface PlayerSummary {
  id: string;
  tracking_id: number;
  team_label: string | null;
  team_color: string | null;
  total_distance_m: number | null;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  avg_x: number | null;
  avg_y: number | null;
  pct_defensive_third: number | null;
  pct_middle_third: number | null;
  pct_attacking_third: number | null;
  is_goalkeeper: boolean;
  is_referee: boolean;
}

export interface TrackingPoint {
  frame: number;
  timestamp_s: number;
  pitch_x: number | null;
  pitch_y: number | null;
  speed_kmh: number | null;
}

export interface PlayerDetail {
  player: PlayerSummary;
  trajectory: TrackingPoint[];
  heatmap_data: number[][];
  speed_series: { timestamp_s: number; speed_kmh: number }[];
}

export interface TeamMetrics {
  team_label: string;
  display_name: string;
  color_hex: string;
  avg_width_m: number | null;
  avg_depth_m: number | null;
  avg_defensive_line_m: number | null;
  avg_compactness_m: number | null;
  total_distance_m: number | null;
  estimated_possession_pct: number | null;
  current_formation: string | null;
}

export interface TacticalEvent {
  id: string;
  timestamp_s: number;
  frame: number;
  event_type: string;
  team_label: string | null;
  value: string | null;
  confidence: number | null;
}

export interface MatchAnalytics {
  match_id: string;
  teams: TeamMetrics[];
  players: PlayerSummary[];
  tactical_events: TacticalEvent[];
  possession_timeline: { timestamp_s: number; frame: number; team_label: string | null }[];
  formation_timeline: { timestamp_s: number; team_label: string; formation: string }[];
  summary: {
    text: string;
    teams: TeamMetrics[];
  };
}

export interface FrameSnapshot {
  frame: number;
  timestamp_s: number;
  players: {
    player_id: string;
    tracking_id: number;
    pitch_x: number | null;
    pitch_y: number | null;
    team_label: string | null;
    team_color: string;
    is_goalkeeper: boolean;
    is_referee: boolean;
  }[];
  ball: { pitch_x: number; pitch_y: number } | null;
  formation_a: string | null;
  formation_b: string | null;
  tactical_phase: string | null;
  possession: string | null;
}

export interface CalibrationPoint {
  img_x: number;
  img_y: number;
  pitch_x: number;
  pitch_y: number;
}

export interface CalibrationResponse {
  success: boolean;
  message: string;
  homography_matrix: number[][] | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  compute_device: string;
  gpu: string | null;
}
