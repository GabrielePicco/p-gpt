// The model pipeline as a node/edge diagram: vocab -> embed -> attention ->
// mlp -> logits. Edges pulse left-to-right on every landed training step.

import { useMemo } from "react";

const LAYERS: { name: string; n: number; shown: number }[] = [
  { name: "vocab", n: 27, shown: 14 },
  { name: "embed", n: 16, shown: 16 },
  { name: "attn", n: 16, shown: 16 },
  { name: "mlp", n: 64, shown: 20 },
  { name: "logits", n: 27, shown: 14 },
];

const W = 300;
const H = 290;
const PAD_Y = 14;

function seeded(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export default function Network({ pulseKey, phase }: { pulseKey: number; phase: number }) {
  const { nodes, edges } = useMemo(() => {
    const rnd = seeded(7);
    const colX = (i: number) => 24 + (i * (W - 48)) / (LAYERS.length - 1);
    const nodes: { x: number; y: number; r: number; layer: number }[] = [];
    const layerNodes: number[][] = [];
    LAYERS.forEach((layer, li) => {
      const ids: number[] = [];
      for (let k = 0; k < layer.shown; k++) {
        const y = PAD_Y + ((k + 0.5) * (H - 2 * PAD_Y)) / layer.shown;
        const r = 2.2 + rnd() * 2.6;
        ids.push(nodes.length);
        nodes.push({ x: colX(li), y, r, layer: li });
      }
      layerNodes.push(ids);
    });
    const edges: { a: number; b: number; w: number }[] = [];
    for (let li = 0; li + 1 < layerNodes.length; li++) {
      for (const a of layerNodes[li]) {
        // a sparse fan-out keeps it legible
        const fan = 1 + Math.floor(rnd() * 3);
        for (let f = 0; f < fan; f++) {
          const b = layerNodes[li + 1][Math.floor(rnd() * layerNodes[li + 1].length)];
          edges.push({ a, b, w: 0.35 + rnd() * 0.65 });
        }
      }
    }
    return { nodes, edges };
  }, []);

  // Which layer is "hot" right now, from the split-training phase.
  const hot = phase === 1 ? [1, 2, 3] : phase === 2 ? [3, 2, 1] : phase === 3 ? [0, 4] : [];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="network" key={pulseKey}>
      <defs>
        <linearGradient id="edge" x1="0" x2="1">
          <stop offset="0" stopColor="#6e5cff" />
          <stop offset="1" stopColor="#7fdcff" />
        </linearGradient>
      </defs>
      {edges.map((e, i) => {
        const a = nodes[e.a];
        const b = nodes[e.b];
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke="url(#edge)"
            strokeWidth={0.6 + e.w * 0.9}
            strokeOpacity={0.18 + e.w * 0.25}
            className="edge"
            style={{ animationDelay: `${(a.layer * 120 + (i % 7) * 30)}ms` }}
          />
        );
      })}
      {nodes.map((n, i) => {
        const glow = hot.includes(n.layer);
        return (
          <circle
            key={i}
            cx={n.x}
            cy={n.y}
            r={n.r}
            className={`node l${n.layer}${glow ? " hot" : ""}`}
          />
        );
      })}
    </svg>
  );
}
