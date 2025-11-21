import React, { useEffect, useState, useRef } from "react";
import io from "socket.io-client";

export default function FirewallSimulator() {
  const [socket, setSocket] = useState(null);
  const [log, setLog] = useState([]);
  const intervalRef = useRef(null);

  useEffect(() => {
    const s = io("http://localhost:8000");
    setSocket(s);

    s.on("firewall_decision", (msg) => {
      const { action, ip } = msg;
      const actionText =
        action === 0 ? "🟢 ALLOW" : action === 1 ? "🟡 RATE-LIMIT" : "🔴 BLOCK";
      const text = `${actionText} – IP ${ip ?? "unknown"}`;

      setLog((prev) => [text, ...prev]);
    });

    return () => s.disconnect();
  }, []);

  const generatePacket = () => [
    Math.random() * 2000,
    Math.floor(Math.random() * 50),
    Math.random(),
    Math.random()
  ];

  const generateIp = () =>
    Array.from({ length: 4 }, () => Math.floor(Math.random() * 256)).join(".");

  const startSim = () => {
    intervalRef.current = setInterval(() => {
      socket.emit("packet", { features: generatePacket(), ip: generateIp() });
    }, 400);
  };

  const stopSim = () => clearInterval(intervalRef.current);

  return (
    <div style={{ padding: 20, fontFamily: "Arial" }}>
      <h1>🔥 RL Firewall Simulation</h1>

      <button onClick={startSim} style={{ marginRight: 10 }}>
        Start Simulation
      </button>
      <button onClick={stopSim}>Stop</button>

      <div
        style={{
          marginTop: 20,
          height: 300,
          overflowY: "scroll",
          background: "#222",
          padding: 10,
          color: "white",
          borderRadius: 8
        }}
      >
        {log.map((entry, i) => (
          <div key={i}>{entry}</div>
        ))}
      </div>
    </div>
  );
}
