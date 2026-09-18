import React, { useEffect, useRef } from 'react';

interface HeatmapProps {
  data: number[][];         // [grid_y][grid_x] values 0..1
  teamColor?: string;
  className?: string;
  opacity?: number;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  teamColor = '#3b82f6',
  className = '',
  opacity = 0.75,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    const W = canvas.width;
    const H = canvas.height;
    const cellW = W / cols;
    const cellH = H / rows;

    ctx.clearRect(0, 0, W, H);

    // Parse team color to RGB
    const r = parseInt(teamColor.slice(1, 3), 16);
    const g = parseInt(teamColor.slice(3, 5), 16);
    const b = parseInt(teamColor.slice(5, 7), 16);

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const v = data[row][col];
        if (v < 0.01) continue;
        const a = v * opacity;
        ctx.fillStyle = `rgba(${r},${g},${b},${a.toFixed(3)})`;
        ctx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
      }
    }

    // Apply Gaussian blur via CSS for smooth look
    canvas.style.filter = 'blur(8px)';
  }, [data, teamColor, opacity]);

  return (
    <canvas
      ref={canvasRef}
      width={520}
      height={338}
      className={`absolute inset-0 w-full h-full rounded-lg ${className}`}
      style={{ mixBlendMode: 'screen' }}
    />
  );
};
