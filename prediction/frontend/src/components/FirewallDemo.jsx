// frontend/src/components/FirewallDemo.jsx
// Cybersecurity Dark Dashboard — Auto Mode Only (Visible Attack Waves)
// Frontend sends sequence history + current to backend /predict_seq every second
// Backend base: import.meta.env.VITE_API_BASE || http://127.0.0.1:8000
// Requires: recharts (npm i recharts)

import React, { useEffect, useMemo, useState, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// local file (uploaded) — convert to URL for image usage if needed
const UPLOADED_IMAGE = "file:///mnt/data/012f61c3-55a1-4f20-8331-32654895e802.png";

const actionLabel = (a) => {
  if (a === 2) return { text: "BLOCK", color: "#ef4444" };
  if (a === 1) return { text: "RATE-LIMIT", color: "#f59e0b" };
  return { text: "ALLOW", color: "#10b981" };
};

// Utility: format time hh:mm:ss
const nowLabel = () => {
  const d = new Date();
  return d.toLocaleTimeString();
};

// Simulation engine helpers
function sampleNormal(mean, std) {
  // Box-Muller
  const u1 = Math.random() || 1e-6;
  const u2 = Math.random() || 1e-6;
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return Math.max(0, Math.round(mean + z0 * std));
}

export default function FirewallDemo() {
  // --- UI & state ---
  const [running, setRunning] = useState(false);
  const [pps, setPps] = useState(120);
  const [uniqueIps, setUniqueIps] = useState(6);
  const [synRatio, setSynRatio] = useState(0.08);
  const [latestDecision, setLatestDecision] = useState(null); // {action, suspicious, ip, ts}
  const [history, setHistory] = useState([]); // decision history for UI
  const [errorMsg, setErrorMsg] = useState(null);

  // sequence buffer for LSTM input
  const SEQ_LEN = 10;
  const [historySeq, setHistorySeq] = useState([]); // each item [pps, uniq, syn]

  // chart time-series of suspicious score for last N points
  const [tsSeries, setTsSeries] = useState([]); // {time, suspicious}

  // ref for interval id
  const timerRef = useRef(null);

  // Attack wave state (internal to simulation)
  const waveRef = useRef({
    mode: "normal", // normal | ramp | attack | recover
    tick: 0,
    // attack intensity 0..1
    intensity: 0,
    // next attack in ticks
    nextAttackIn: Math.floor(8 + Math.random() * 10),
  });

  // Seed initial plausible values
  useEffect(() => {
    setPps(sampleNormal(120, 30));
    setUniqueIps(Math.max(1, sampleNormal(6, 2)));
    setSynRatio(Number((0.06 + Math.random() * 0.05).toFixed(3)));
    // populate historySeq with plausible values
    const init = [];
    for (let i = 0; i < SEQ_LEN; i++) {
      init.push([
        sampleNormal(120, 30),
        Math.max(1, sampleNormal(6, 2)),
        Number((0.06 + Math.random() * 0.05).toFixed(3)),
      ]);
    }
    setHistorySeq(init);
  }, []);

  // simulation step: mutate base values according to waveRef
  const simulateStep = () => {
    const wr = waveRef.current;
    wr.tick += 1;

    // trigger an attack occasionally
    if (wr.nextAttackIn <= 0 && wr.mode === "normal") {
      wr.mode = "ramp";
      wr.intensity = Math.random() * 0.6 + 0.3; // 0.3 .. 0.9
      wr.rampTicks = 3 + Math.floor(Math.random() * 3);
      wr.attackTicks = 3 + Math.floor(Math.random() * 5);
      wr.recoverTicks = 3 + Math.floor(Math.random() * 4);
      wr.tick = 0;
    }

    // decrease countdown
    if (wr.mode === "normal") {
      wr.nextAttackIn -= 1;
    }

    // baseline drift
    let basePps = sampleNormal(120, 25);
    let baseUnique = Math.max(1, sampleNormal(6, 2));
    let baseSyn = Math.min(
      0.9,
      Math.max(0.01, Number((0.05 + Math.random() * 0.06).toFixed(3)))
    );

    // adjust depending on wave mode
    if (wr.mode === "ramp") {
      const frac = Math.min(1, wr.tick / Math.max(1, wr.rampTicks));
      const mult = 1 + frac * (1.0 + wr.intensity * 2); // grows
      basePps = Math.round(basePps * mult + wr.intensity * 400);
      baseSyn = Math.min(1, baseSyn + frac * (0.3 + wr.intensity * 0.5));
      baseUnique = Math.round(baseUnique * (1 + frac * 2.5));
      if (wr.tick >= wr.rampTicks) {
        wr.mode = "attack";
        wr.tick = 0;
      }
    } else if (wr.mode === "attack") {
      const mult = 2.5 + wr.intensity * 3.0;
      basePps = Math.round(basePps * mult + 1000 * wr.intensity);
      baseSyn = Math.min(1, baseSyn + 0.4 + wr.intensity * 0.5);
      baseUnique = Math.round(baseUnique * (2 + wr.intensity * 3));
      if (wr.tick >= wr.attackTicks) {
        wr.mode = "recover";
        wr.tick = 0;
      }
    } else if (wr.mode === "recover") {
      const frac = Math.min(1, wr.tick / Math.max(1, wr.recoverTicks));
      const mult = 1 + (1 - frac) * (1.5 + wr.intensity * 2);
      basePps = Math.round(basePps * mult);
      baseSyn = Math.min(1, baseSyn + (1 - frac) * 0.25);
      baseUnique = Math.round(baseUnique * (1 + (1 - frac) * 2));
      if (wr.tick >= wr.recoverTicks) {
        wr.mode = "normal";
        wr.tick = 0;
        wr.intensity = 0;
        wr.nextAttackIn = 8 + Math.floor(Math.random() * 12);
      }
    } else {
      // normal small jitter
      basePps = Math.round(basePps + Math.random() * 30 - 15);
    }

    // small randomness to make values varied
    basePps = Math.max(1, Math.round(basePps + (Math.random() * 50 - 25)));
    baseUnique = Math.max(1, Math.round(baseUnique + Math.random() * 6 - 3));
    baseSyn = Number(
      Math.min(0.999, Math.max(0.0, baseSyn + (Math.random() * 0.05 - 0.025))).toFixed(3)
    );

    return { pps: basePps, unique: baseUnique, syn: baseSyn, wave: wr.mode };
  };

  // perform one simulation step: generate triple, send to backend, update UI
  const runStep = async () => {
    const sim = simulateStep();
    const triple = [sim.pps, sim.unique, sim.syn];

    // update the sequence buffer locally (we send to server the latest buffer including triple)
    const sentHistory = [...historySeq, triple].slice(-SEQ_LEN);

    // Build payload
    const payload = {
      history: sentHistory,
      current: triple,
      ip: `192.168.${Math.floor(Math.random() * 254)}.${Math.floor(Math.random() * 254)}`,
    };

    // Optimistically update UI values (so chart & display respond instantly)
    setPps(sim.pps);
    setUniqueIps(sim.unique);
    setSynRatio(sim.syn);

    // Call backend predict_seq
    try {
      const res = await fetch(`${API_BASE}/predict_seq`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Server ${res.status}`);
      }
      const data = await res.json();

      // Record decision entry
      const entry = {
        ts: nowLabel(),
        ip: payload.ip,
        current: triple,
        action: data.action,
        suspicious: data.suspicious ?? null,
        wave: sim.wave,
      };

      // update UI arrays
      setHistory((h) => [entry, ...h].slice(0, 200));
      setHistorySeq(sentHistory);
      setLatestDecision({ ...entry });
      setTsSeries((s) => [...s, { time: nowLabel(), suspicious: data.suspicious ?? 0 }].slice(-80));
    } catch (err) {
      setErrorMsg(String(err));
      // still update local sequence and UI so demo continues
      const entry = {
        ts: nowLabel(),
        ip: payload.ip,
        current: triple,
        action: -1,
        suspicious: null,
        wave: sim.wave,
      };
      setHistory((h) => [entry, ...h].slice(0, 200));
      setHistorySeq(sentHistory);
      setLatestDecision({ ...entry });
      // ensure chart still advances
      setTsSeries((s) => [...s, { time: nowLabel(), suspicious: 0 }].slice(-80));
    }
  };

  // Start/Stop simulation
  const startSimulation = () => {
    if (running) return;
    setErrorMsg(null);
    setRunning(true);
    // start interval (1s)
    timerRef.current = setInterval(() => {
      runStep();
    }, 1000);
  };

  const stopSimulation = () => {
    setRunning(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Chart data for bar counts
  const barData = useMemo(() => {
    const counts = { ALLOW: 0, "RATE-LIMIT": 0, BLOCK: 0, UNKNOWN: 0 };
    history.forEach((h) => {
      if (h.action === 0) counts.ALLOW++;
      else if (h.action === 1) counts["RATE-LIMIT"]++;
      else if (h.action === 2) counts.BLOCK++;
      else counts.UNKNOWN++;
    });
    return [
      { name: "ALLOW", count: counts.ALLOW, color: "#10b981" },
      { name: "RATE-LIMIT", count: counts["RATE-LIMIT"], color: "#f59e0b" },
      { name: "BLOCK", count: counts.BLOCK, color: "#ef4444" },
    ];
  }, [history]);

  // format small status card for latest
  const LatestCard = ({ latest }) => {
    const info = latest ? actionLabel(latest.action) : null;
    return (
      <div
        style={{
          background: "#061018",
          padding: 14,
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.03)",
        }}
      >
        <div style={{ color: "#94a3b8", fontSize: 13 }}>Latest Decision</div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 10 }}>
          <div
            style={{
              width: 70,
              height: 70,
              borderRadius: 12,
              background: info ? info.color : "#374151",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontWeight: 800,
              fontSize: 16,
            }}
          >
            {info ? info.text : "--"}
          </div>
          <div>
            <div style={{ color: "#e6eef6", fontWeight: 700 }}>{latest ? latest.ip : "—"}</div>
            <div style={{ color: "#94a3b8", fontSize: 12 }}>{latest ? latest.ts : ""}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>Wave: {latest ? latest.wave : "—"}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>
              Suspicious: {latest && latest.suspicious != null ? Number(latest.suspicious).toFixed(3) : "—"}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // === LAYOUT: make full-width container (changes from previous maxWidth center) ===
  return (
    <div style={{ width: "100vw", minHeight: "100vh", background: "linear-gradient(180deg,#041018 0%, #071624 100%)", padding: 24, boxSizing: "border-box" }}>
      {/* Full-width header bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, gap: 12 }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ color: "#e6f0ff", margin: 0, fontSize: 32 }}>Autonomous AI Firewall — Live Demo</h1>
          <div style={{ color: "#94a3b8", marginTop: 6 }}>Auto mode: Visible Attack Waves · LSTM (suspicious) + XGBoost (decision)</div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 12 }}>
          <button
            onClick={startSimulation}
            disabled={running}
            style={{
              background: "#06b6d4",
              color: "#022023",
              border: "none",
              padding: "10px 14px",
              borderRadius: 10,
              fontWeight: 800,
              cursor: running ? "not-allowed" : "pointer",
            }}
          >
            Start Simulation
          </button>
          <button
            onClick={stopSimulation}
            disabled={!running}
            style={{
              background: "#374151",
              color: "#e6eef6",
              border: "none",
              padding: "10px 14px",
              borderRadius: 10,
              fontWeight: 700,
              cursor: !running ? "not-allowed" : "pointer",
            }}
          >
            Stop
          </button>
        </div>
      </div>

      {/* Main grid stretches full width */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 20, alignItems: "start" }}>
        {/* Left: wide column */}
        <div>
          {/* Live telemetry card */}
          <div style={{ display: "flex", gap: 14, marginBottom: 14 }}>
            <div style={{ flex: 1, background: "#07121a", padding: 18, borderRadius: 12 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Realtime Telemetry</div>
              <div style={{ display: "flex", gap: 24, marginTop: 12 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>Packets/sec</div>
                  <div style={{ color: "#e6eef6", fontSize: 28, fontWeight: 800 }}>{pps}</div>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>Unique IPs</div>
                  <div style={{ color: "#e6eef6", fontSize: 28, fontWeight: 800 }}>{uniqueIps}</div>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>SYN Ratio</div>
                  <div style={{ color: "#e6eef6", fontSize: 28, fontWeight: 800 }}>{synRatio.toFixed(3)}</div>
                </div>
              </div>
            </div>

            <div style={{ width: 260 }}>
              <LatestCard latest={latestDecision} />
            </div>
          </div>

          {/* Suspicious trend chart */}
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12, marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Suspicious Score (recent)</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>{tsSeries.length} samples</div>
            </div>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tsSeries}>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} stroke="#9ca3af" />
                  <Tooltip />
                  <Line type="monotone" dataKey="suspicious" stroke="#06b6d4" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Live decision feed */}
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Live Decisions Feed</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>{history.length} total</div>
            </div>

            <div style={{ marginTop: 10, maxHeight: "50vh", overflowY: "auto", paddingRight: 6 }}>
              {history.length === 0 && <div style={{ color: "#94a3b8" }}>Simulation stopped — start to see activity.</div>}
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {history.map((h, i) => (
                  <li key={i} style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", background: "#0b1520", padding: 12, borderRadius: 10, marginBottom: 10 }}>
                    <div>
                      <div style={{ color: "#e6eef6", fontWeight: 800 }}>{h.ip} <span style={{ color: "#9ca3af", fontSize: 12, marginLeft: 8 }}>{h.ts}</span></div>
                      <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 8 }}>pps: {h.current[0]}, ips: {h.current[1]}, syn: {h.current[2].toFixed(3)}</div>
                      <div style={{ color: "#9ca3af", fontSize: 12, marginTop: 6 }}>wave: {h.wave}</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                      <div style={{ padding: "8px 12px", borderRadius: 8, background: actionLabel(h.action).color, color: "#fff", fontWeight: 900 }}>{actionLabel(h.action).text}</div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>susp: {h.suspicious != null ? Number(h.suspicious).toFixed(3) : "—"}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Right: summary + chart (fixed column) */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ color: "#9ca3af", fontSize: 13 }}>Overview</div>
            <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ background: "#08121a", padding: 12, borderRadius: 8 }}>
                <div style={{ color: "#9ca3af", fontSize: 12 }}>Total Decisions</div>
                <div style={{ color: "#e6eef6", fontSize: 20, fontWeight: 800, marginTop: 8 }}>{history.length}</div>
              </div>
              <div style={{ background: "#08121a", padding: 12, borderRadius: 8 }}>
                <div style={{ color: "#9ca3af", fontSize: 12 }}>Current Wave</div>
                <div style={{ color: "#e6eef6", fontSize: 18, fontWeight: 700, marginTop: 8 }}>{(latestDecision && latestDecision.wave) || "—"}</div>
              </div>
            </div>
          </div>

          <div style={{ background: "#07121a", padding: 14, borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Decision Counts</div>
              <div style={{ color: "#9ca3af", fontSize: 12 }}>Last {history.length}</div>
            </div>

            <div style={{ width: "100%", height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData}>
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count">
                    {barData.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          
        </div>
      </div>

      {/* Footer / errors (full width) */}
      <div style={{ marginTop: 20 }}>
        {errorMsg && (
          <div style={{ background: "#fee2e2", color: "#991b1b", padding: 10, borderRadius: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>Error: {errorMsg}</div>
              <button onClick={() => setErrorMsg(null)} style={{ textDecoration: "underline", background: "transparent", border: "none" }}>Dismiss</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
