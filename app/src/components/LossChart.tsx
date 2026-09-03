// Training loss over steps: single-series line on the chart surface with a
// crosshair + tooltip hover layer.

import { useEffect, useRef, useState } from "react";

const CSS = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export interface LossPoint {
  step: number;
  loss: number;
}

export default function LossChart({ points }: { points: LossPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = 200;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.height = `${h}px`;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);

    g.fillStyle = CSS("--surface-1");
    g.fillRect(0, 0, w, h);
    if (points.length < 2) {
      g.fillStyle = CSS("--text-muted");
      g.font = "12px JetBrains Mono";
      g.fillText("waiting for training steps…", 12, h / 2);
      return;
    }

    const pad = { l: 38, r: 10, t: 10, b: 20 };
    const xs = points.map((p) => p.step);
    const ys = points.map((p) => p.loss);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const yMin = Math.max(0, Math.min(...ys) - 0.15);
    const yMax = Math.max(...ys) + 0.15;
    const px = (s: number) => pad.l + ((s - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
    const py = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);

    // Recessive grid + y labels.
    g.strokeStyle = CSS("--border-strong");
    g.globalAlpha = 0.35;
    g.lineWidth = 1;
    g.fillStyle = CSS("--text-muted");
    g.font = "10px JetBrains Mono";
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const v = yMin + ((yMax - yMin) * i) / gridLines;
      const y = py(v);
      g.beginPath();
      g.moveTo(pad.l, y);
      g.lineTo(w - pad.r, y);
      g.stroke();
      g.globalAlpha = 1;
      g.fillText(v.toFixed(2), 4, y + 3);
      g.globalAlpha = 0.35;
    }
    g.globalAlpha = 1;

    // Raw per-step loss, recessive; a smoothed trend on top carries the story.
    g.strokeStyle = CSS("--series-1");
    g.lineJoin = "round";
    g.lineWidth = 1;
    g.globalAlpha = 0.35;
    g.beginPath();
    points.forEach((p, i) => {
      if (i === 0) g.moveTo(px(p.step), py(p.loss));
      else g.lineTo(px(p.step), py(p.loss));
    });
    g.stroke();
    g.globalAlpha = 1;
    g.lineWidth = 2.2;
    g.beginPath();
    let ema = points[0].loss;
    points.forEach((p, i) => {
      ema += (p.loss - ema) * 0.12;
      if (i === 0) g.moveTo(px(p.step), py(ema));
      else g.lineTo(px(p.step), py(ema));
    });
    g.stroke();

    // Crosshair.
    if (hoverX !== null && hoverX >= pad.l && hoverX <= w - pad.r) {
      let nearest = points[0];
      let best = Infinity;
      for (const p of points) {
        const d = Math.abs(px(p.step) - hoverX);
        if (d < best) {
          best = d;
          nearest = p;
        }
      }
      const cx = px(nearest.step);
      const cy = py(nearest.loss);
      g.strokeStyle = CSS("--text-muted");
      g.globalAlpha = 0.5;
      g.beginPath();
      g.moveTo(cx, pad.t);
      g.lineTo(cx, h - pad.b);
      g.stroke();
      g.globalAlpha = 1;
      g.fillStyle = CSS("--series-1");
      g.beginPath();
      g.arc(cx, cy, 4, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = CSS("--surface-1");
      g.lineWidth = 2;
      g.stroke();
    }
  }, [points, hoverX]);

  return (
    <>
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          setHoverX(x);
          if (points.length >= 2) {
            const xs = points.map((p) => p.step);
            const x0 = Math.min(...xs);
            const x1 = Math.max(...xs);
            const frac = (x - 38) / Math.max(1, rect.width - 48);
            const step = Math.round(x0 + frac * (x1 - x0));
            let nearest = points[0];
            for (const p of points) {
              if (Math.abs(p.step - step) < Math.abs(nearest.step - step)) nearest = p;
            }
            setTip({
              x: e.clientX,
              y: e.clientY,
              text: `step ${nearest.step}\nloss ${nearest.loss.toFixed(4)}`,
            });
          }
        }}
        onMouseLeave={() => {
          setHoverX(null);
          setTip(null);
        }}
      />
      {tip && (
        <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.text}
        </div>
      )}
    </>
  );
}
