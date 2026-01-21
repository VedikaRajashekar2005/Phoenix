// frontend/src/components/FirewallDemo.jsx
// Cybersecurity Dark Dashboard – Auto Mode Only (Visible Attack Waves)
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
  const [rps, setRps] = useState(50);
  const [bandwidth, setBandwidth] = useState(5);
  const [latestDecision, setLatestDecision] = useState(null); // {action, suspicious, ip, ts}
  const [history, setHistory] = useState([]); // decision history for UI
  const [errorMsg, setErrorMsg] = useState(null);

  // sequence buffer for LSTM input - NOW 6 FEATURES
  const SEQ_LEN = 10;
  const [historySeq, setHistorySeq] = useState([]); // each item: 6-feature vector

  // chart time-series of suspicious score for last N points
  const [tsSeries, setTsSeries] = useState([]); // {time, suspicious}

  // ref for interval id
  const timerRef = useRef(null);

  // Persistent Wave State
  const waveRef = useRef({
    pattern: "normal",
    ticksLeft: 6
  });

  // Seed initial plausible values with 6 features ONCE
  // [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]
  const [initialized, setInitialized] = useState(false);
  
  useEffect(() => {
    if (initialized) return;
    
    setPps(sampleNormal(120, 30));
    setUniqueIps(Math.max(1, sampleNormal(6, 2)));
    setSynRatio(Number((0.06 + Math.random() * 0.05).toFixed(3)));
    setRps(sampleNormal(50, 15));
    setBandwidth(sampleNormal(5, 2));
    
    // populate historySeq with plausible 6-feature vectors
    const init = [];
    for (let i = 0; i < SEQ_LEN; i++) {
      init.push([
        sampleNormal(120, 30),                            // pps
        Math.max(1, sampleNormal(6, 2)),                  // unique_ips
        Number((0.06 + Math.random() * 0.05).toFixed(3)), // syn_ratio
        sampleNormal(50, 15),                             // rps
        sampleNormal(5, 2),                               // bandwidth (Mbps)
        1.0,                                              // always_on flag
      ]);
    }
    setHistorySeq(init);
    setInitialized(true);
  }, [initialized]);

  // =====================================================
  // Persistent continuous wave simulator (no disappearing)
  // =====================================================
  const simulateStep = () => {
    const wr = waveRef.current;

    // If wave finished, choose a new one
    if (wr.ticksLeft <= 0) {
      const patterns = ["normal","protocol","application","volumetric","mixed"];
      wr.pattern = patterns[Math.floor(Math.random() * patterns.length)];
      wr.ticksLeft = 4 + Math.floor(Math.random() * 5); // 4–8 seconds
    }

    wr.ticksLeft -= 1;
    const pattern = wr.pattern;

    // --- Base normal ---
    let basePps = sampleNormal(120, 25);
    let baseUnique = Math.max(1, sampleNormal(6, 2));
    let baseSyn = Math.min(
      0.9,
      Math.max(0.01, Number((0.05 + Math.random() * 0.06).toFixed(3)))
    );
    let baseRps = sampleNormal(50, 15);
    let baseBw = sampleNormal(5, 2);

    const intensity = Math.random() * 0.6 + 0.4;

    // --- Protocol ---
    if (pattern === "protocol" || pattern === "mixed") {
      basePps = Math.round(basePps + 1200 * intensity + Math.random()*800);
      baseSyn = Math.min(1, baseSyn + 0.4 + intensity * 0.5);
      baseUnique = Math.round(baseUnique * (2 + intensity * 3));
    }

    // --- Application ---
    if (pattern === "application" || pattern === "mixed") {
      baseRps = Math.round(baseRps + 700 * intensity + Math.random()*1200);
      baseUnique = Math.round(baseUnique * (1.5 + intensity * 2));
    }

    // --- Volumetric ---
    if (pattern === "volumetric" || pattern === "mixed") {
      baseBw = Math.round(baseBw + 200 * intensity + Math.random()*400);
      basePps = Math.round(basePps + 600 * intensity);
    }

    // --- Noise ---
    basePps = Math.max(1, basePps + Math.round(Math.random()*40 - 20));
    baseUnique = Math.max(1, baseUnique + Math.round(Math.random()*4 - 2));
    baseSyn = Number(Math.min(0.999, Math.max(0.0, baseSyn + (Math.random()*0.04 - 0.02))).toFixed(3));
    baseRps = Math.max(1, baseRps + Math.round(Math.random()*15 - 7));
    baseBw = Math.max(1, baseBw + Math.round(Math.random()*3 - 1));

    return {
      pps: basePps,
      unique: baseUnique,
      syn: baseSyn,
      rps: baseRps,
      bw: baseBw,
      wave: pattern
    };
  };

  // perform one simulation step: generate 6-feature vector, send to backend, update UI
  const runStep = async () => {
    const sim = simulateStep();
    
    // Create 6-feature vector matching training:
    // [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]
    const sixFeatures = [
      sim.pps,      // 0: pps
      sim.unique,   // 1: unique_ips
      sim.syn,      // 2: syn_ratio
      sim.rps,      // 3: rps
      sim.bw,       // 4: bandwidth (Mbps)
      1.0,          // 5: always_on flag
    ];

    // update the sequence buffer locally (we send to server the latest buffer including sixFeatures)
    const sentHistory = [...historySeq, sixFeatures].slice(-SEQ_LEN);

    // Build payload
    const payload = {
      history: sentHistory,
      current: sixFeatures,
      ip: `192.168.${Math.floor(Math.random() * 254)}.${Math.floor(Math.random() * 254)}`,
    };

    // Optimistically update UI values (so chart & display respond instantly)
    setPps(sim.pps);
    setUniqueIps(sim.unique);
    setSynRatio(sim.syn);
    setRps(sim.rps);
    setBandwidth(sim.bw);

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
        current: sixFeatures,
        action: data.action,
        suspicious: data.suspicious ?? null,
        wave: sim.wave,
      };

      // update UI arrays
      setHistory((h) => [entry, ...h].slice(0, 200));
      setHistorySeq(sentHistory);
      setLatestDecision({ ...entry });
      // Keep wave on screen always (no slice), simply append
      setTsSeries((s) => [...s, { time: nowLabel(), suspicious: data.suspicious ?? 0 }]);
      setErrorMsg(null); // clear errors on success
    } catch (err) {
      setErrorMsg(String(err));
      // still update local sequence and UI so demo continues
      const entry = {
        ts: nowLabel(),
        ip: payload.ip,
        current: sixFeatures,
        action: -1,
        suspicious: null,
        wave: sim.wave,
      };
      setHistory((h) => [entry, ...h].slice(0, 200));
      setHistorySeq(sentHistory);
      setLatestDecision({ ...entry });
      // ensure chart still advances
      setTsSeries((s) => [...s, { time: nowLabel(), suspicious: 0 }]);
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
            <div style={{ color: "#e6eef6", fontWeight: 700 }}>{latest ? latest.ip : "–"}</div>
            <div style={{ color: "#94a3b8", fontSize: 12 }}>{latest ? latest.ts : ""}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>Wave: {latest ? latest.wave : "–"}</div>
            <div style={{ color: "#94a3b8", marginTop: 6 }}>
              Suspicious: {latest && latest.suspicious != null ? Number(latest.suspicious).toFixed(3) : "–"}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // === LAYOUT: make full-width container ===
  return (
    <div style={{ width: "100vw", minHeight: "100vh", background: "linear-gradient(180deg,#041018 0%, #071624 100%)", padding: 24, boxSizing: "border-box" }}>
      {/* Full-width header bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, gap: 12 }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ color: "#e6f0ff", margin: 0, fontSize: 32 }}>Autonomous AI Firewall – Live Demo</h1>
          <div style={{ color: "#94a3b8", marginTop: 6 }}>Auto mode: Visible Attack Waves · LSTM (suspicious) + XGBoost (decision)</div>
          <div style={{ color: "#64748b", marginTop: 4, fontSize: 12 }}>6 Features: [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]</div>
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
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, marginTop: 12 }}>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>PPS</div>
                  <div style={{ color: "#e6eef6", fontSize: 24, fontWeight: 800 }}>{pps}</div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>Unique IPs</div>
                  <div style={{ color: "#e6eef6", fontSize: 24, fontWeight: 800 }}>{uniqueIps}</div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>SYN Ratio</div>
                  <div style={{ color: "#e6eef6", fontSize: 24, fontWeight: 800 }}>{synRatio.toFixed(3)}</div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>RPS</div>
                  <div style={{ color: "#e6eef6", fontSize: 24, fontWeight: 800 }}>{rps}</div>
                </div>
                <div>
                  <div style={{ color: "#9ca3af", fontSize: 12 }}>BW (Mbps)</div>
                  <div style={{ color: "#e6eef6", fontSize: 24, fontWeight: 800 }}>{bandwidth}</div>
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
                  <Line 
                    type="monotone" 
                    dataKey="suspicious" 
                    stroke="#06b6d4" 
                    strokeWidth={2} 
                    dot={false}
                    isAnimationActive={false} // Prevents "slideshow" wipe effect on update
                  />
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
              {history.length === 0 && <div style={{ color: "#94a3b8" }}>Simulation stopped – start to see activity.</div>}
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {history.map((h, i) => (
                  <li key={i} style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", background: "#0b1520", padding: 12, borderRadius: 10, marginBottom: 10 }}>
                    <div>
                      <div style={{ color: "#e6eef6", fontWeight: 800 }}>{h.ip} <span style={{ color: "#9ca3af", fontSize: 12, marginLeft: 8 }}>{h.ts}</span></div>
                      <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 8 }}>
                        pps: {h.current[0]}, ips: {h.current[1]}, syn: {h.current[2].toFixed(3)}, rps: {h.current[3]}, bw: {h.current[4]}
                      </div>
                      <div style={{ color: "#9ca3af", fontSize: 12, marginTop: 6 }}>wave: {h.wave}</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                      <div style={{ padding: "8px 12px", borderRadius: 8, background: actionLabel(h.action).color, color: "#fff", fontWeight: 900 }}>{actionLabel(h.action).text}</div>
                      <div style={{ color: "#94a3b8", fontSize: 12 }}>susp: {h.suspicious != null ? Number(h.suspicious).toFixed(3) : "–"}</div>
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
                <div style={{ color: "#e6eef6", fontSize: 18, fontWeight: 700, marginTop: 8 }}>{(latestDecision && latestDecision.wave) || "–"}</div>
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
              <button onClick={() => setErrorMsg(null)} style={{ textDecoration: "underline", background: "transparent", border: "none", cursor: "pointer" }}>Dismiss</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}