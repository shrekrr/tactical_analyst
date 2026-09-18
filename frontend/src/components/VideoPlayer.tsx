import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, Volume2, VolumeX, SkipBack, SkipForward } from 'lucide-react';

interface VideoPlayerProps {
  src: string;
  onTimeUpdate?: (currentTime: number, duration: number) => void;
  seekTo?: number;
  className?: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  onTimeUpdate,
  seekTo,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (seekTo !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTo;
    }
  }, [seekTo]);

  const handleTimeUpdate = () => {
    const v = videoRef.current;
    if (!v) return;
    const ct = v.currentTime;
    const dur = v.duration || 1;
    setCurrentTime(ct);
    setProgress((ct / dur) * 100);
    onTimeUpdate?.(ct, dur);
  };

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const v = videoRef.current;
    if (!v) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    v.currentTime = pct * v.duration;
  };

  const skip = (seconds: number) => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, Math.min(v.currentTime + seconds, v.duration));
  };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`card overflow-hidden group ${className}`}>
      <video
        ref={videoRef}
        src={src}
        className="w-full aspect-video bg-black object-contain"
        muted={muted}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onClick={togglePlay}
        style={{ cursor: 'pointer' }}
      />

      {/* Controls */}
      <div className="px-4 pb-3 pt-2 space-y-2 bg-surface-800">
        {/* Progress bar */}
        <div
          className="h-1.5 bg-surface-600 rounded-full cursor-pointer relative"
          onClick={handleSeek}
        >
          <div
            className="absolute inset-y-0 left-0 bg-blue-500 rounded-full transition-none"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Buttons + time */}
        <div className="flex items-center gap-3">
          <button onClick={() => skip(-10)} className="text-slate-400 hover:text-slate-200 transition-colors">
            <SkipBack size={15} />
          </button>
          <button
            onClick={togglePlay}
            className="w-8 h-8 rounded-full bg-blue-600 hover:bg-blue-500 flex items-center justify-center transition-colors"
          >
            {playing ? <Pause size={14} className="text-white" /> : <Play size={14} className="text-white ml-0.5" />}
          </button>
          <button onClick={() => skip(10)} className="text-slate-400 hover:text-slate-200 transition-colors">
            <SkipForward size={15} />
          </button>
          <button
            onClick={() => setMuted(!muted)}
            className="text-slate-400 hover:text-slate-200 transition-colors ml-1"
          >
            {muted ? <VolumeX size={15} /> : <Volume2 size={15} />}
          </button>
          <span className="text-xs text-slate-400 font-mono ml-auto">
            {fmt(currentTime)} / {fmt(duration)}
          </span>
        </div>
      </div>
    </div>
  );
};
