// All 4,192 parameters as one dense column of squares — one square, one
// Q32.32 weight, straight from the rollup. Cyan is negative, pink positive,
// near-zero stays dark, so structure emerging out of noise is visible.

import { useEffect, useRef, useState } from "react";

const COLS = 64;
const N = 4192;
const ROWS = Math.ceil(N / COLS);
const CELL = 7;
const GAP = 1;
const LABEL_W = 30;

// Tensor boundaries in the flat layout, for row labels.
const TENSORS: [string, number][] = [
  ["wte", 0],
  ["wpe", 432],
  ["wq", 688],
  ["wk", 944],
  ["wv", 1200],
  ["wo", 1456],
  ["w1", 1712],
  ["w2", 2736],
  ["lm", 3760],
];

function color(v: number): string {
  // Diverging around a dark midpoint: cyan (-) / pink (+); |v| ~0.6 saturates.
  const t = Math.max(-1, Math.min(1, v / 0.6));
  const u = Math.abs(t);
  const mid = [24, 22, 48];
  const pole = t < 0 ? [127, 220, 255] : [255, 122, 200];
  const m = (a: number, b: number) => Math.round(a + (b - a) * Math.pow(u, 0.75));
  return `rgb(${m(mid[0], pole[0])},${m(mid[1], pole[1])},${m(mid[2], pole[2])})`;
}

export default function WeightsMap({ weights }: { weights: Float64Array | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const width = LABEL_W + COLS * (CELL + GAP);
  const height = ROWS * (CELL + GAP);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, height);
    g.font = "9px JetBrains Mono";
    g.fillStyle = "#6f6a9c";
    for (const [name, start] of TENSORS) {
      g.fillText(name, 0, Math.floor(start / COLS) * (CELL + GAP) + 8);
    }
    if (!weights) return;
    for (let i = 0; i < N; i++) {
      const r = Math.floor(i / COLS);
      const c = i % COLS;
      g.fillStyle = color(weights[i]);
      g.fillRect(LABEL_W + c * (CELL + GAP), r * (CELL + GAP), CELL, CELL);
    }
  }, [weights, width, height]);

  return (
    <div className="weights-wrap">
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          if (!weights) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const c = Math.floor((e.clientX - rect.left - LABEL_W) / (CELL + GAP));
          const r = Math.floor((e.clientY - rect.top) / (CELL + GAP));
          const i = r * COLS + c;
          if (c < 0 || c >= COLS || i < 0 || i >= N) {
            setTip(null);
            return;
          }
          const tensor = [...TENSORS].reverse().find(([, s]) => i >= s)!;
          setTip({
            x: e.clientX,
            y: e.clientY,
            text: `${tensor[0]}[${i - tensor[1]}]  #${i}\n${weights[i].toFixed(5)}`,
          });
        }}
        onMouseLeave={() => setTip(null)}
      />
      {tip && (
        <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}
