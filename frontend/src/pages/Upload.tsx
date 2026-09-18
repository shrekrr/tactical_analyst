import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileVideo, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { uploadVideo, startAnalysis } from '../services/api';

const MAX_SIZE_GB = 2;
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/avi'];
const ALLOWED_EXT = ['.mp4', '.mov', '.avi', '.mkv'];

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

type Stage = 'idle' | 'uploading' | 'starting' | 'done' | 'error';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [stage, setStage] = useState<Stage>('idle');
  const [uploadPct, setUploadPct] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [matchId, setMatchId] = useState<string | null>(null);
  const [sampleFps, setSampleFps] = useState(5);

  const validateFile = (f: File): string | null => {
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) {
      return `Unsupported format. Please upload: ${ALLOWED_EXT.join(', ')}`;
    }
    if (f.size > MAX_SIZE_GB * 1024 * 1024 * 1024) {
      return `File exceeds ${MAX_SIZE_GB} GB limit.`;
    }
    return null;
  };

  const handleFile = (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }
    setError(null);
    setFile(f);
    setStage('idle');
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setStage('uploading');
    setError(null);
    setUploadPct(0);
    try {
      const res = await uploadVideo(file, (pct) => setUploadPct(pct));
      setMatchId(res.match_id);
      setStage('starting');
      await startAnalysis(res.match_id, sampleFps);
      setStage('done');
      setTimeout(() => navigate(`/match/${res.match_id}`), 1500);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Upload failed. Please try again.');
      setStage('error');
    }
  };

  return (
    <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center p-8">
      {/* Header */}
      <div className="mb-10 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center">
            <span className="text-white font-bold text-lg">E</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100">EPL Tactical Analyst</h1>
        </div>
        <p className="text-sm text-slate-500 max-w-md">
          Upload a football match video to extract player tracking, formations, heatmaps, and tactical insights.
        </p>
      </div>

      {/* Upload card */}
      <div className="w-full max-w-xl">
        <div className="card p-6 space-y-6">
          {/* Drop zone */}
          <label
            htmlFor="video-input"
            className={`block border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
              ${dragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-slate-400 hover:bg-surface-700/50'}
              ${file ? 'border-emerald-500/50 bg-emerald-500/5' : ''}
            `}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              id="video-input"
              type="file"
              accept={ALLOWED_EXT.join(',')}
              className="hidden"
              onChange={handleInputChange}
            />
            {file ? (
              <div className="flex flex-col items-center gap-3">
                <CheckCircle className="text-emerald-400" size={36} />
                <div>
                  <div className="font-medium text-slate-100 text-sm">{file.name}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{formatBytes(file.size)}</div>
                </div>
                <span className="text-xs text-emerald-400">Ready to upload</span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="w-14 h-14 rounded-full bg-surface-600 flex items-center justify-center">
                  <Upload className="text-slate-400" size={24} />
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-300">Drag & drop your match video</div>
                  <div className="text-xs text-slate-500 mt-1">or click to browse</div>
                </div>
                <div className="flex gap-2 flex-wrap justify-center">
                  {ALLOWED_EXT.map(ext => (
                    <span key={ext} className="badge bg-surface-600 text-slate-400 text-xs">{ext.toUpperCase()}</span>
                  ))}
                </div>
                <div className="text-xs text-slate-600">Maximum size: {MAX_SIZE_GB} GB</div>
              </div>
            )}
          </label>

          {/* Settings */}
          {file && stage === 'idle' && (
            <div className="space-y-3">
              <label className="stat-label">Analysis Frame Rate</label>
              <div className="flex gap-2">
                {[1, 5, 10, 25].map(fps => (
                  <button
                    key={fps}
                    onClick={() => setSampleFps(fps)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-all ${
                      sampleFps === fps
                        ? 'bg-blue-600/30 border-blue-500 text-blue-300'
                        : 'bg-surface-700 border-slate-600 text-slate-400 hover:border-slate-500'
                    }`}
                  >
                    {fps} FPS
                  </button>
                ))}
              </div>
              <p className="text-xs text-slate-500">
                Higher FPS = more accurate tracking, longer processing time.{' '}
                <strong className="text-slate-400">5 FPS recommended.</strong>
              </p>
            </div>
          )}

          {/* Progress */}
          {stage === 'uploading' && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Uploading…</span>
                <span>{uploadPct}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
              </div>
            </div>
          )}

          {stage === 'starting' && (
            <div className="flex items-center gap-3 text-sm text-blue-400">
              <Loader2 className="animate-spin" size={16} />
              Starting analysis pipeline…
            </div>
          )}

          {stage === 'done' && (
            <div className="flex items-center gap-3 text-sm text-emerald-400">
              <CheckCircle size={16} />
              Analysis queued! Redirecting to dashboard…
            </div>
          )}

          {error && (
            <div className="flex items-start gap-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          {/* Upload button */}
          <button
            onClick={handleUpload}
            disabled={!file || stage === 'uploading' || stage === 'starting' || stage === 'done'}
            className="w-full btn-primary justify-center py-3 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {stage === 'uploading' && <Loader2 className="animate-spin" size={16} />}
            {stage === 'starting' && <Loader2 className="animate-spin" size={16} />}
            {stage === 'idle' && file && <FileVideo size={16} />}
            {stage === 'idle' && file ? 'Upload & Analyze' : 'Select a video first'}
          </button>
        </div>

        {/* Back link */}
        <div className="text-center mt-4">
          <button
            onClick={() => navigate('/')}
            className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
          >
            ← View all matches
          </button>
        </div>
      </div>
    </div>
  );
};
