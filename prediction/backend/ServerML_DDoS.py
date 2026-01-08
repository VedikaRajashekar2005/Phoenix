import os
import time
import threading
from collections import defaultdict

import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from scapy.all import sniff
    SCAPY_AVAILABLE = True
except:
    SCAPY_AVAILABLE = False

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print(f"[INFO] Model directory: {MODEL_DIR}")

# Load models with error handling
model = None
scaler = None

try:
    model_path = os.path.join(MODEL_DIR, "xgb.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"[OK] Loaded model from {model_path}")
    else:
        print(f"[WARN] Model not found at {model_path}")
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Loaded scaler from {scaler_path}")
    else:
        print(f"[WARN] Scaler not found at {scaler_path}")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")

# ---------------- FLASK ----------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------------- STATS ----------------

packet_sizes = []
packet_count = 0
byte_count = 0
lock = threading.Lock()

WINDOW_SEC = 1.0
latest_prediction = {"status": "initializing"}

# ---------------- PACKET CAPTURE ----------------

def packet_handler(packet):
    global packet_count, byte_count

    size = len(packet)

    with lock:
        packet_sizes.append(size)
        packet_count += 1
        byte_count += size

def start_packet_capture():
    """Start packet capture in background (optional)"""
    if not SCAPY_AVAILABLE:
        print("[WARN] Scapy not available, skipping packet capture")
        return
    
    try:
        print("[INFO] Starting packet capture...")
        sniff(prn=packet_handler, store=False)
    except Exception as e:
        print(f"[WARN] Packet capture failed (may need admin): {e}")

# ---------------- FEATURE EXTRACTION ----------------

def extract_features():
    global packet_sizes, packet_count, byte_count

    with lock:
        sizes = packet_sizes.copy()
        count = packet_count
        bytes_ = byte_count

        packet_sizes = []
        packet_count = 0
        byte_count = 0

    if count == 0:
        return None

    pkt_len_mean = np.mean(sizes) if sizes else 64.0
    pkt_len_std = np.std(sizes) if sizes else 10.0
    pkt_rate = count / WINDOW_SEC
    byte_rate = bytes_ / WINDOW_SEC

    features = np.array([[
        pkt_len_mean,
        pkt_len_std,
        pkt_rate,
        byte_rate
    ]])

    if scaler is None:
        return features
    return scaler.transform(features)

# ---------------- PREDICTION LOOP ----------------

def prediction_loop():
    global latest_prediction

    while True:
        time.sleep(WINDOW_SEC)

        features = extract_features()
        if features is None:
            continue

        pred = model.predict(features)[0]

        latest_prediction = {
            "ddos_detected": bool(pred),
            "timestamp": time.time(),
            "features": {
                "pkt_len_mean": float(features[0][0]),
                "pkt_len_std": float(features[0][1]),
                "pkt_rate": float(features[0][2]),
                "byte_rate": float(features[0][3])
            }
        }

# ---------------- API ----------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "AI Firewall Backend",
        "endpoints": ["/health", "/predict", "/predict_seq"]
    })

@app.route("/predict", methods=["GET"])
def predict():
    return jsonify(latest_prediction)

@app.route("/predict_seq", methods=["POST"])
def predict_seq():
    try:
        data = request.json
        history = data.get("history", [])
        current = data.get("current", [])
        ip = data.get("ip", "unknown")
        
        if len(current) >= 3:
            pps = float(current[0])
            unique_ips = float(current[1])
            syn_ratio = float(current[2])
            
            # Build features for XGBoost model
            # Model expects: [pkt_len_mean, pkt_len_std, pkt_rate, byte_rate]
            pkt_len_mean = 60.0 + (syn_ratio * 20)  # Adjust based on SYN ratio
            pkt_len_std = 8.0
            pkt_rate = pps
            byte_rate = pps * (pkt_len_mean + 20)  # More realistic byte calculation
            
            features = np.array([[pkt_len_mean, pkt_len_std, pkt_rate, byte_rate]])
            
            pred = 0
            suspicious = 0.0
            
            if model and scaler:
                try:
                    features_scaled = scaler.transform(features)
                    pred = int(model.predict(features_scaled)[0])
                    suspicious = float(model.predict_proba(features_scaled)[0][1]) if hasattr(model, 'predict_proba') else float(pred)
                except Exception as e:
                    print(f"[WARN] Prediction error: {e}")
                    # Fallback: use heuristic based on SYN ratio
                    suspicious = min(0.99, syn_ratio * 2)
                    pred = 1 if syn_ratio > 0.15 else 0
            else:
                # Fallback heuristic if model not loaded
                suspicious = min(0.99, syn_ratio * 2 + (pps / 1000))
                pred = 1 if (syn_ratio > 0.15 or pps > 500) else 0
        else:
            pred = 0
            suspicious = 0.0
        
        # Map prediction to action: 0=ALLOW, 1=RATE-LIMIT, 2=BLOCK
        if pred > 0:
            action = 2 if suspicious > 0.5 else 1
        else:
            action = 0
        
        return jsonify({
            "action": action,
            "suspicious": float(suspicious),
            "ip": ip,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"[ERROR] /predict_seq error: {e}")
        return jsonify({"error": str(e), "action": 0, "suspicious": 0.0}), 400

@app.route("/traffic", methods=["GET"])
def get_traffic():
    """Get current traffic statistics"""
    global packet_sizes, packet_count, byte_count
    
    with lock:
        sizes = packet_sizes.copy()
        count = packet_count
        bytes_ = byte_count
    
    if count == 0:    ### CAUSE FOR DATA ON FRONTEND TO BE 0 -- needs windows admin access to get packet count
        return jsonify({
            "pps": 0,
            "byte_rate": 0,
            "unique_ips": 0, 
            "syn_ratio": 0.0,
            "packet_count": 0
        })
    
    pkt_len_mean = np.mean(sizes) if sizes else 64
    pkt_len_std = np.std(sizes) if sizes else 10
    pkt_rate = count / WINDOW_SEC
    byte_rate = bytes_ / WINDOW_SEC
    
    # Estimate SYN ratio from packet patterns (simplified: use packet size variance)
    syn_ratio = min(0.99, pkt_len_std / 100.0) if pkt_len_std > 0 else 0.05
    
    return jsonify({
        "pps": int(pkt_rate),
        "byte_rate": int(byte_rate),
        "unique_ips": int(count / 10) if count > 0 else 0,
        "syn_ratio": round(float(syn_ratio), 3),
        "packet_count": count,
        "packet_len_mean": round(float(pkt_len_mean), 2),
        "packet_len_std": round(float(pkt_len_std), 2)
    })

# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI Firewall Backend Starting")
    print("="*50)
    print(f"API running on http://0.0.0.0:8000")
    print(f"Test with: http://localhost:8000/health")
    print("="*50)
    print("[INFO] Starting packet capture (requires admin privileges)...")
    print("="*50 + "\n")
    
    # Start real packet capture only
    threading.Thread(
        target=start_packet_capture,
        daemon=True
    ).start()

    # Start prediction loop
    threading.Thread(
        target=prediction_loop,
        daemon=True
    ).start()

    app.run(host="0.0.0.0", port=8000, debug=False)

