// The attention lens: the four heads' causal attention matrices for the last
// generated name, read straight from the on-chain generation workspace.
// Sequential single-hue ramp (magnitude 0..1).

import { useEffect, useRef, useState } from "react";
import { BLOCK, N_HEAD } from "../lib/chain";

const CELL = 9;
const GAP = 18;

function seqColor(v: number): string {
  // Dark surface -> bright blue, single hue.
  const u = Math.max(0, Math.min(1, v));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * u);
  return `rgb(${lerp(26, 120)},${lerp(26, 175)},${lerp(30, 255)})`;
}

export default function Attention({
  att,
  tokens,
}: {
  att: Float64Array | null;
  tokens: string; // the generated text (position t consumed tokens[t-1]... display only)
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const n = Math.min(BLOCK, Math.max(2, tokens.length + 1));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const width = N_HEAD * (n * CELL + GAP) - GAP;
    const height = n * CELL + 16;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, height);
    if (!att) {
      g.fillStyle = "#8b8a80";
      g.font = "11px JetBrains Mono";
      g.fillText("generate a name to see the model attend", 0, 20);
      return;
    }
    g.font = "10px JetBrains Mono";
    for (let h = 0; h < N_HEAD; h++) {
      const ox = h * (n * CELL + GAP);
      g.fillStyle = "#8b8a80";
      g.fillText(`head ${h}`, ox, 10);
      for (let t = 0; t < n; t++) {
        for (let s = 0; s <= t; s++) {
          const v = att[(t * N_HEAD + h) * BLOCK + s];
          g.fillStyle = seqColor(v);
          g.fillRect(ox + s * CELL, 14 + t * CELL, CELL - 1, CELL - 1);
        }
      }
    }
  }, [att, n]);

  return (
    <div style={{ overflowX: "auto" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "auto" }}
        onMouseMove={(e) => {
          if (!att) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top - 14;
          const h = Math.floor(mx / (n * CELL + GAP));
          const s = Math.floor((mx - h * (n * CELL + GAP)) / CELL);
          const t = Math.floor(my / CELL);
          if (h >= 0 && h < N_HEAD && t >= 0 && t < n && s >= 0 && s <= t) {
            const v = att[(t * N_HEAD + h) * BLOCK + s];
            setTip({
              x: e.clientX,
              y: e.clientY,
              text: `head ${h}  t=${t} ← s=${s}\nweight ${v.toFixed(3)}`,
            });
          } else {
            setTip(null);
          }
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
