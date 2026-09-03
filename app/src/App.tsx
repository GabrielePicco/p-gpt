import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import LossChart, { LossPoint } from "./components/LossChart";
import WeightsMap from "./components/WeightsMap";
import Attention from "./components/Attention";
import Network from "./components/Network";
import {
  fetchAttention,
  sendCheckpoint,
  sendContribute,
  sendGenerate,
  watchBase,
  watchEr,
  BaseState,
  LiveModel,
  BASE_URL,
  ER_URL,
} from "./lib/chain";
import { GenEntry, PROGRAM_ID } from "./lib/pgpt";

const PHASES = ["PICK DOC", "FORWARD", "BACKWARD", "ADAM"];

export default function App() {
  const [live, setLive] = useState<LiveModel | null>(null);
  const [baseState, setBaseState] = useState<BaseState | null>(null);
  const [gens, setGens] = useState<GenEntry[]>([]);
  const [genTotal, setGenTotal] = useState(0n);
  const [losses, setLosses] = useState<LossPoint[]>([]);
  const [att, setAtt] = useState<Float64Array | null>(null);
  const [prefix, setPrefix] = useState("");
  const [contrib, setContrib] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [pulseKey, setPulseKey] = useState(0);

  const window_ = useRef<{ t: number; step: number }[]>([]);
  const [stepsPerSec, setStepsPerSec] = useState(0);
  const lastSeen = useRef(Date.now());

  const onModel = useCallback((m: LiveModel) => {
    lastSeen.current = Date.now();
    setLive((prev) => {
      if (!prev || m.header.step !== prev.header.step) {
        const now = Date.now();
        const w = window_.current;
        w.push({ t: now, step: Number(m.header.step) });
        while (w.length > 2 && now - w[0].t > 30_000) w.shift();
        if (w.length >= 2) {
          const dt = (now - w[0].t) / 1000;
          if (dt > 1) setStepsPerSec((Number(m.header.step) - w[0].step) / dt);
        }
        setPulseKey((k) => k + 1);
      }
      return m;
    });
    const step = Number(m.header.step);
    const ring = m.header.lossRing;
    setLosses(
      ring.map((loss, i) => ({ step: step - ring.length + i, loss })).filter((p) => p.step >= 0),
    );
  }, []);

  const onGen = useCallback((entries: GenEntry[], total: bigint) => {
    setGens(entries.slice().reverse());
    setGenTotal(total);
  }, []);

  useEffect(() => watchEr(onModel, onGen), [onModel, onGen]);
  useEffect(() => watchBase(setBaseState), []);

  const doGenerate = async () => {
    setBusy("generate");
    try {
      await sendGenerate(prefix, 0.5);
      setAtt(await fetchAttention());
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(null);
    }
  };
  const doContribute = async () => {
    setBusy("contribute");
    try {
      await sendContribute(contrib);
      setContrib("");
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(null);
    }
  };
  const doCheckpoint = async () => {
    setBusy("checkpoint");
    try {
      await sendCheckpoint();
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(null);
    }
  };

  const h = live?.header;
  const training = stepsPerSec > 0 && Date.now() - lastSeen.current < 20_000;
  const last = gens[0];
  const doc = h?.currentDoc ?? "";
  const cursor = h?.phase === 1 ? h.phaseCursor : h?.phase === 2 ? h.phaseCursor : -1;
  const lag = useMemo(
    () => (h && baseState?.header ? Number(h.step - baseState.header.step) : null),
    [h, baseState],
  );

  return (
    <>
      <div className="top">
        <div className="logo">
          <i />
          <i />
        </div>
        <div className="title">
          <h1>
            SOLANA <span>/</span> microGPT
          </h1>
          <p>A complete GPT — training and inference — inside one Solana program</p>
        </div>
        <div className={`pill-live${training ? "" : " idle"}`}>
          <b /> {training ? "LIVE TRAINING" : "STANDBY"}
        </div>
      </div>

      <div className="cols">
        {/* ── column 1: what it is learning right now ── */}
        <div className="col">
          <div className="cap">Learning right now</div>
          <div className="hero">
            {h ? `#${Number(h.step).toLocaleString()}` : "—"}
          </div>
          <div className="sub">
            <span>gradient step</span>
            <span>{h ? PHASES[h.phase] ?? "" : ""}</span>
          </div>
          <div className="docbox">
            <div className="doc-tiles">
              {Array.from({ length: 16 }, (_, i) => {
                const ch = doc[i];
                const cls = ch ? (i === cursor ? "cur" : "on") : "";
                return (
                  <span key={i} className={cls}>
                    {ch ?? ""}
                  </span>
                );
              })}
            </div>
          </div>
          <div className="chip">one name / 15 tokens / ≤16 positions</div>
          <div className="kpis">
            <div className="kpi">
              <div className="cap">loss</div>
              <div className="v">{h ? h.lastLoss.toFixed(3) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="cap">loss ema</div>
              <div className="v">{h ? h.lossEma.toFixed(3) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="cap">speed</div>
              <div className="v">
                {stepsPerSec > 0 ? stepsPerSec.toFixed(2) : "0.00"}
                <small>steps/s</small>
              </div>
            </div>
            <div className="kpi">
              <div className="cap">txs / step</div>
              <div className="v">
                ~{doc ? 2 * Math.min(16, doc.length + 1) + 18 : 34}
              </div>
            </div>
          </div>
          <div className="chip dim">start ≈ ln 27 = 3.296 — every step is a transaction</div>
        </div>

        {/* ── column 2: the model ── */}
        <div className="col">
          <div className="cap">Model pipeline</div>
          <div className="hero">
            27 <small>→</small> 16 <small>→</small> 64 <small>→</small> 27
          </div>
          <div className="sub">
            <span>vocab</span>
            <span>embed</span>
            <span>mlp</span>
            <span>logits</span>
          </div>
          <Network pulseKey={pulseKey} phase={h?.phase ?? 0} />
          <div className="pred">
            <div className="cap">Prediction</div>
            <div className="name">{last ? last.name : "—"}</div>
            <div className="meta">
              <b>sampled</b>
              <br />
              step {last ? last.step.toString() : "—"}
              <br />
              temp 0.5
            </div>
          </div>
          <div className="ask">
            <input
              type="text"
              placeholder="prefix…"
              value={prefix}
              maxLength={15}
              onChange={(e) => setPrefix(e.target.value.toLowerCase().replace(/[^a-z]/g, ""))}
            />
            <button onClick={doGenerate} disabled={busy !== null}>
              {busy === "generate" ? "…" : "generate"}
            </button>
          </div>
          <div className="chip dim">prefix → 16 tokens → Q32.32 → one transaction</div>
        </div>

        {/* ── column 3: the weights ── */}
        <div className="col">
          <div className="cap">Training weights</div>
          <div className="hero">
            4,192 <small>×</small> 8 BYTES
          </div>
          <div className="sub">
            <span>one square = one Q32.32 weight / live from the rollup</span>
          </div>
          <WeightsMap weights={live?.weights ?? null} />
          <div className="legend">
            <span>
              <i style={{ background: "#7fdcff" }} />
              negative
            </span>
            <span>
              <i style={{ background: "#ff7ac8" }} />
              positive
            </span>
            <span>
              <i style={{ background: "#181630" }} />
              ≈ zero
            </span>
          </div>
        </div>
      </div>

      <div className="row2">
        <div className="card">
          <h2>
            Training loss <small>cross-entropy per step</small>
          </h2>
          <LossChart points={losses} />
          <h2 style={{ marginTop: 18 }}>
            Attention lens <small>the 4 heads reading the last name</small>
          </h2>
          <Attention att={att} tokens={last?.name ?? ""} />
        </div>

        <div className="card">
          <h2>
            The babble feed <small>{genTotal.toString()} names</small>
          </h2>
          <div className="babble">
            {gens.length === 0 && <div className="hint">no generations yet — ask it for a name</div>}
            {gens.map((g, i) => (
              <div className="babble-entry" key={`${g.step}-${i}`}>
                <span className="step">step {g.step.toString()}</span>
                <span className="name">{g.name || "∅"}</span>
              </div>
            ))}
          </div>
          <div className="ask" style={{ marginTop: 14 }}>
            <input
              type="text"
              placeholder="teach it your name"
              value={contrib}
              maxLength={15}
              onChange={(e) => setContrib(e.target.value.toLowerCase().replace(/[^a-z]/g, ""))}
            />
            <button className="ghost" onClick={doContribute} disabled={busy !== null || !contrib}>
              contribute
            </button>
          </div>
          <p className="hint">contributed names join the community dataset — trained on every 8th step</p>
        </div>

        <div className="card">
          <h2>Two layers, one model</h2>
          <div className="layers">
            <div className="layer">
              <div className="n">
                <span className={`dot${baseState ? " live" : ""}`} /> Solana base layer
              </div>
              <div className="kv">
                <div>endpoint</div>
                <div>{BASE_URL.replace("http://", "")}</div>
                <div>model</div>
                <div>{baseState?.delegated ? "delegated" : "resident"}</div>
                <div>checkpoint</div>
                <div>{baseState?.header ? `step ${baseState.header.step}` : "—"}</div>
                <div>lag</div>
                <div>{lag === null ? "—" : `${lag} steps`}</div>
              </div>
            </div>
            <div className="layer">
              <div className="n">
                <span className={`dot${live ? " live" : ""}`} /> Ephemeral rollup
              </div>
              <div className="kv">
                <div>endpoint</div>
                <div>{ER_URL.replace("http://", "")}</div>
                <div>live step</div>
                <div>{h ? h.step.toString() : "—"}</div>
                <div>crank</div>
                <div>{training ? "running" : "idle"}</div>
              </div>
              <button
                className="ghost"
                style={{ marginTop: 10, width: "100%" }}
                onClick={doCheckpoint}
                disabled={busy !== null}
              >
                {busy === "checkpoint" ? "committing…" : "checkpoint to base"}
              </button>
            </div>
          </div>
          <p className="hint" style={{ marginTop: 10 }}>
            program {PROGRAM_ID.toBase58().slice(0, 8)}… · no backend: the rollup websocket is the
            data source
          </p>
        </div>
      </div>

      <footer>
        Karpathy's microGPT (4,192 params) — 1 layer, 16-dim, 4 heads — trained by the chain
        itself: forward, backward and Adam are on-chain instructions in Q32.32 fixed-point,
        bit-exact replayable. A perpetual crank on a MagicBlock Ephemeral Rollup fires
        <code> TrainMicro </code> every 100ms; checkpoints commit the model image to Solana.
      </footer>
    </>
  );
}
