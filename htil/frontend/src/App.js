import React, { useState, useEffect, useCallback } from 'react';
import { 
  Shield, 
  Activity, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Server, 
  RefreshCw, 
  BarChart3,
  Terminal,
  Network,
  TrendingUp
} from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5002';

const ACTIONS = {
  0: { label: 'ALLOW', color: 'var(--color-allow)', icon: <CheckCircle size={18} /> },
  1: { label: 'RATE LIMIT', color: 'var(--color-limit)', icon: <AlertTriangle size={18} /> },
  2: { label: 'BLOCK', color: 'var(--color-block)', icon: <XCircle size={18} /> }
};

function App() {
  const [samples, setSamples] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [correctionMode, setCorrectionMode] = useState(false);
  const [notification, setNotification] = useState(null);

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      const [samplesRes, statsRes] = await Promise.all([
        fetch(`${API_URL}/load_samples`),
        fetch(`${API_URL}/stats`)
      ]);

      const samplesData = await samplesRes.json();
      const statsData = await statsRes.json();

      if (samplesData.success) setSamples(samplesData.samples);
      if (statsData.success) setStats(statsData.stats);
    } catch (error) {
      showNotification('Error connecting to Firewall API', 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh stats every 30s
    return () => clearInterval(interval);
  }, [fetchData]);

  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  const handleReview = async (isCorrect, correctedAction = null) => {
    if (samples.length === 0) return;
    
    setProcessing(true);
    const currentSample = samples[0];

    try {
      const response = await fetch(`${API_URL}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: currentSample,
          is_correct: isCorrect,
          corrected_action: correctedAction
        })
      });

      const result = await response.json();

      if (result.success) {
        showNotification(result.message, isCorrect ? 'success' : 'warning');
        // Remove reviewed sample from local state
        setSamples(prev => prev.slice(1));
        // Refresh stats to show updated counts
        const statsRes = await fetch(`${API_URL}/stats`);
        const statsData = await statsRes.json();
        if (statsData.success) setStats(statsData.stats);
        setCorrectionMode(false);
      }
    } catch (error) {
      showNotification('Failed to submit review', 'error');
    } finally {
      setProcessing(false);
    }
  };

  const triggerRetraining = async () => {
    try {
      const response = await fetch(`${API_URL}/trigger_retraining`, { method: 'POST' });
      const result = await response.json();
      showNotification(result.message);
      fetchData();
    } catch (error) {
      showNotification('Failed to trigger retraining', 'error');
    }
  };

  if (loading) return <div className="loading-screen"><Activity className="spin" /> Initializing Dashboard...</div>;

  const currentSample = samples[0];
  const networkParams = currentSample?.network_parameters || {};
  const paramAnalysis = currentSample?.parameter_analysis || {};

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <Shield className="brand-icon" />
          <div>
            <h1>SENTINEL</h1>
            <span className="subtitle">DDoS Human Review</span>
          </div>
        </div>

        <div className="stats-container">
          <div className="stat-card">
            <span className="stat-label">Model Accuracy</span>
            <div className="stat-value">
              {stats?.accuracy}%
              <div className="progress-bar">
                <div className="fill" style={{ width: `${stats?.accuracy}%` }}></div>
              </div>
            </div>
          </div>

          <div className="stat-grid">
            <div className="mini-stat">
              <label>Reviewed</label>
              <span>{stats?.total_reviewed}</span>
            </div>
            <div className="mini-stat">
              <label>Pending</label>
              <span>{samples.length}</span>
            </div>
            <div className="mini-stat success">
              <label>Correct</label>
              <span>{stats?.correct_predictions}</span>
            </div>
            <div className="mini-stat error">
              <label>Incorrect</label>
              <span>{stats?.incorrect_predictions}</span>
            </div>
          </div>

          <div className="learning-status">
            <h3><RefreshCw size={16} /> Continuous Learning</h3>
            <div className="status-row">
              <span>Feedback Buffer:</span>
              <span className="mono">{stats?.pending_feedback.count} / {stats?.pending_feedback.threshold}</span>
            </div>
            <div className="status-row">
              <span>Status:</span>
              <span className={stats?.retraining_in_progress ? "status-badge active" : "status-badge"}>
                {stats?.retraining_in_progress ? "RETRAINING..." : "IDLE"}
              </span>
            </div>
            <button 
              className="retrain-btn" 
              onClick={triggerRetraining}
              disabled={stats?.retraining_in_progress}
            >
              {stats?.retraining_in_progress ? "System Busy" : "Force Retrain"}
            </button>
          </div>
        </div>

        <div className="system-logs">
          <div className="log-header"><Terminal size={14} /> System Log</div>
          <div className="log-content">
            {stats?.retraining_in_progress && <div className="log-entry warning">{'>'} Neural Network Updating...</div>}
            <div className="log-entry">{'>'} System Active</div>
            <div className="log-entry">{'>'} Monitoring Port 8000</div>
            <div className="log-entry">{'>'} {samples.length} anomalies detected</div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header>
          <h2>Anomaly Inspection</h2>
          <div className="connection-status">
            <div className="dot"></div> Live Connection
          </div>
        </header>

        {notification && (
          <div className={`notification ${notification.type}`}>
            {notification.type === 'success' ? <CheckCircle size={20}/> : <AlertTriangle size={20}/>}
            {notification.message}
          </div>
        )}

        {currentSample ? (
          <div className="review-interface">
            {/* Prediction Header */}
            <div className="prediction-card">
              <div className="prediction-info">
                <span className="label">Source IP</span>
                <span className="value mono ip">{currentSample.ip}</span>
              </div>
              <div className="prediction-info">
                <span className="label">AI Recommendation</span>
                <div className="action-badge" style={{ borderColor: ACTIONS[currentSample.predicted_action].color }}>
                  {ACTIONS[currentSample.predicted_action].icon}
                  <span style={{ color: ACTIONS[currentSample.predicted_action].color }}>
                    {ACTIONS[currentSample.predicted_action].label}
                  </span>
                </div>
              </div>
              <div className="prediction-info">
                <span className="label">Suspicion Score</span>
                <span className="value mono">{(currentSample.predicted_suspicious * 100).toFixed(2)}%</span>
              </div>
            </div>

            {/* Network Parameters */}
            <div className="network-parameters">
              <h3><Network size={18} /> Network Traffic Metrics</h3>
              <div className="params-grid">
                <div className="param-item">
                  <span className="param-label">Packets/sec</span>
                  <span className="param-value mono">{networkParams.packets_per_second?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">Bytes/sec</span>
                  <span className="param-value mono">{networkParams.bytes_per_second?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">SYN Ratio</span>
                  <span className="param-value mono">{networkParams.syn_ratio?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">Unique Ports</span>
                  <span className="param-value mono">{networkParams.unique_ports || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">Avg Packet Size</span>
                  <span className="param-value mono">{networkParams.packet_size_avg?.toFixed(2) || 'N/A'} B</span>
                </div>
                <div className="param-item">
                  <span className="param-label">Entropy</span>
                  <span className="param-value mono">{networkParams.entropy?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">TCP Ratio</span>
                  <span className="param-value mono">{networkParams.tcp_ratio?.toFixed(4) || 'N/A'}</span>
                </div>
                <div className="param-item">
                  <span className="param-label">UDP Ratio</span>
                  <span className="param-value mono">{networkParams.udp_ratio?.toFixed(4) || 'N/A'}</span>
                </div>
              </div>

              {networkParams.traffic_spike_ratio && (
                <div className="traffic-spike">
                  <TrendingUp size={16} />
                  <span>Traffic Spike: <strong>{networkParams.traffic_spike_ratio}x</strong> normal levels</span>
                </div>
              )}
            </div>

            {/* Parameter Analysis */}
            {(paramAnalysis.suspicious_indicators?.length > 0 || paramAnalysis.normal_indicators?.length > 0) && (
              <div className="parameter-analysis">
                <h3><AlertTriangle size={18} /> Automated Analysis</h3>
                
                {paramAnalysis.suspicious_indicators?.length > 0 && (
                  <div className="analysis-section suspicious">
                    <span className="analysis-title">⚠️ Suspicious Indicators:</span>
                    <ul className="indicator-list">
                      {paramAnalysis.suspicious_indicators.map((indicator, idx) => (
                        <li key={idx}>{indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {paramAnalysis.normal_indicators?.length > 0 && (
                  <div className="analysis-section normal">
                    <span className="analysis-title">✓ Normal Indicators:</span>
                    <ul className="indicator-list">
                      {paramAnalysis.normal_indicators.map((indicator, idx) => (
                        <li key={idx}>{indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {paramAnalysis.severity_score !== undefined && (
                  <div className="severity-score">
                    <span>Threat Severity:</span>
                    <div className="severity-bar">
                      <div 
                        className="severity-fill" 
                        style={{ 
                          width: `${(paramAnalysis.severity_score / 10) * 100}%`,
                          backgroundColor: paramAnalysis.severity_score > 6 ? 'var(--color-block)' : 
                                          paramAnalysis.severity_score > 3 ? 'var(--color-limit)' : 
                                          'var(--color-allow)'
                        }}
                      />
                    </div>
                    <span className="severity-value">{paramAnalysis.severity_score}/10</span>
                  </div>
                )}
              </div>
            )}

            {/* Traffic Visualization */}
            <div className="traffic-analysis">
              <h3><BarChart3 size={18} /> Traffic Vector Analysis</h3>
              <div className="vector-grid">
                {currentSample.current.map((val, idx) => (
                  <div key={idx} className="vector-item">
                    <span className="v-label">Feat {idx}</span>
                    <div className="v-bar-container">
                      <div 
                        className="v-bar" 
                        style={{ 
                          height: `${Math.min(val * 10, 100)}%`,
                          backgroundColor: val > 5 ? 'var(--color-limit)' : 'var(--accent-blue)' 
                        }} 
                      />
                    </div>
                    <span className="v-val">{val.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Area */}
            <div className="action-area">
              {!correctionMode ? (
                <>
                  <button 
                    className="btn-confirm"
                    onClick={() => handleReview(true)}
                    disabled={processing}
                  >
                    <CheckCircle /> Confirm {ACTIONS[currentSample.predicted_action].label}
                  </button>
                  <button 
                    className="btn-reject"
                    onClick={() => setCorrectionMode(true)}
                    disabled={processing}
                  >
                    <XCircle /> Incorrect Prediction
                  </button>
                </>
              ) : (
                <div className="correction-panel">
                  <span className="correction-title">Select Correct Action:</span>
                  <div className="correction-options">
                    {[0, 1, 2]
                      .filter(a => a !== currentSample.predicted_action)
                      .map(action => (
                        <button
                          key={action}
                          className="btn-option"
                          style={{ '--hover-color': ACTIONS[action].color }}
                          onClick={() => handleReview(false, action)}
                        >
                          {ACTIONS[action].icon} Set to {ACTIONS[action].label}
                        </button>
                    ))}
                    <button className="btn-cancel" onClick={() => setCorrectionMode(false)}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <Server size={64} />
            <h3>All Clear</h3>
            <p>No anomalies pending human review.</p>
            <p className="sub-text">Waiting for new data from Firewall...</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;