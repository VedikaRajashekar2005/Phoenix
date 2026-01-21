from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
SCALER_PATH = os.getenv("SCALER_PATH")
LSTM_PATH   = os.getenv("LSTM_PATH")
XGB_PATH    = os.getenv("XGB_PATH")
INTERMEDIATE_PATH = os.getenv("INTERMEDIATE_JSON_PATH")
DEVICE = torch.device("cpu")

# Feature count - MUST match training
NUM_FEATURES = 6  # [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]

CONFIDENCE_THRESHOLD = 0.65

# ---- FLASK APP ----
app = Flask(__name__)
CORS(app)

# =========================================================
# LSTM Model Definition (Must match training code)
# =========================================================

class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.act(self.fc(last)).squeeze(1)

# =========================================================

scaler = None
lstm = None
xgb_clf = None

# =========================================================
# Load Models
# =========================================================

def try_load():
    global scaler, lstm, xgb_clf
    
    # ---- Load Scaler ----
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Loaded scaler (n_features: {getattr(scaler, 'n_features_in_', 'unknown')})")

    # ---- Load LSTM ----
    if os.path.exists(LSTM_PATH):
        lstm = SuspiciousLSTM()
        lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
        lstm.to(DEVICE)
        lstm.eval()
        print(f"✓ Loaded LSTM (input_dim: {NUM_FEATURES})")

    # ---- Load XGBoost ----
    if os.path.exists(XGB_PATH):
        xgb_clf = joblib.load(XGB_PATH)
        print(f"✓ Loaded XGBoost")

try_load()

# =========================================================
# Utility Functions
# =========================================================

def compute_suspicious_score(full_sequence: np.ndarray) -> float:
    """
    Compute suspicious score using LSTM on the FULL sequence
    
    Args:
        full_sequence: np.ndarray of shape (sequence_length, 6)
                      Features: [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]
    
    Returns:
        float: suspicious score between 0 and 1
    """
    if lstm is None:
        # fallback heuristic using the LAST row (current features)
        last_row = full_sequence[-1]
        pps = last_row[0]
        unique_ips = last_row[1] if len(last_row) > 1 else 0
        syn = last_row[2] if len(last_row) > 2 else 0
        rps = last_row[3] if len(last_row) > 3 else 0
        bw = last_row[4] if len(last_row) > 4 else 0
        
        # Heuristic scoring
        s = 0.0
        s += min(0.3, pps / 1500)       # PPS contributes up to 0.3
        s += min(0.2, bw / 300)          # Bandwidth contributes up to 0.2
        s += min(0.2, rps / 1000)        # RPS contributes up to 0.2
        s += min(0.3, syn)               # SYN ratio contributes up to 0.3
        
        return float(np.clip(s, 0, 1))
    
    # LSTM expects shape: (batch_size, sequence_length, features)
    arr = full_sequence.astype("float32")[None, ...]  # Add batch dimension
    
    print(f"🔍 LSTM Input shape: {arr.shape}")  # DEBUG
    print(f"🔍 LSTM Input (last 2 timesteps):\n{full_sequence[-2:]}")  # DEBUG
    
    with torch.no_grad():
        t = torch.from_numpy(arr).to(DEVICE)
        out = lstm(t).cpu().numpy()
        score = float(out[0])
        
        print(f"🔍 LSTM Raw output: {score:.4f}")  # DEBUG
        return score


def xgb_decision(feature_vec: np.ndarray):
    """
    feature_vec = [6 current features + suspicious_score] = 7 features total
    Features: [pps, unique_ips, syn_ratio, rps, bandwidth, always_on, suspicious_score]
    
    Returns (action, confidence):
        action: 0=Allow, 1=RateLimit, 2=Block
    """

    if xgb_clf is None:
        # fallback rules
        pps = feature_vec[0]
        unique_ips = feature_vec[1]
        syn = feature_vec[2]
        rps = feature_vec[3]
        bw = feature_vec[4]
        susp = feature_vec[-1]

        print(f"🔍 XGB Fallback - PPS: {pps:.0f}, BW: {bw:.0f}, RPS: {rps:.0f}, SYN: {syn:.3f}, SUSP: {susp:.3f}")

        # More aggressive thresholds based on training data
        if pps > 1500 or bw > 300 or rps > 1000 or susp > 0.8:
            print(f"   → BLOCK (pps={pps:.0f} > 1500 OR bw={bw:.0f} > 300 OR rps={rps:.0f} > 1000 OR susp={susp:.3f} > 0.8)")
            return 2, 0.85     # BLOCK
        if pps > 900 or rps > 600 or susp > 0.5:
            print(f"   → RATE LIMIT (pps={pps:.0f} > 900 OR rps={rps:.0f} > 600 OR susp={susp:.3f} > 0.5)")
            return 1, 0.70     # RATE LIMIT
        print(f"   → ALLOW (all thresholds passed)")
        return 0, 0.90         # ALLOW

    X = feature_vec.reshape(1, -1)

    probs = xgb_clf.predict_proba(X)[0]
    
    # Handle models trained with fewer than 3 classes
    n_classes = len(probs)
    if n_classes == 2:
        # Model only knows Allow (0) and RateLimit (1)
        # Map to our 3-class system: [Allow, RateLimit, Block=0]
        full_probs = np.array([probs[0], probs[1], 0.0])
        print(f"⚠️  XGB model only trained on 2 classes (Allow, RateLimit)")
    elif n_classes == 3:
        full_probs = probs
    else:
        # Fallback for unexpected class count
        print(f"❌ Unexpected number of classes: {n_classes}")
        full_probs = np.array([0.8, 0.2, 0.0])
    
    action = int(np.argmax(full_probs))
    confidence = float(full_probs[action])
    
    action_names = {0: "Allow", 1: "RateLimit", 2: "Block"}
    print(f"🔍 XGB Model Prediction:")
    print(f"   Input features: PPS={feature_vec[0]:.0f}, IPs={feature_vec[1]:.0f}, SYN={feature_vec[2]:.3f}, RPS={feature_vec[3]:.0f}, BW={feature_vec[4]:.0f}, Susp={feature_vec[-1]:.3f}")
    print(f"   Probabilities: Allow={full_probs[0]:.3f}, RateLimit={full_probs[1]:.3f}, Block={full_probs[2]:.3f}")
    print(f"   → Decision: {action_names[action]} (confidence: {confidence:.3f})")
    
    return action, confidence


def log_to_intermediate(history, current, ip, action, suspicious, confidence):
    """Log prediction to intermediate.json"""

    needs_review = confidence < CONFIDENCE_THRESHOLD

    entry = {
        "history": history,
        "current": current,
        "ip": ip,
        "predicted_action": int(action),
        "predicted_suspicious": float(suspicious),
        "confidence": float(confidence),
        "needs_review": needs_review,
        "timestamp": datetime.utcnow().isoformat()
    }

    if os.path.exists(INTERMEDIATE_PATH):
        try:
            data = json.load(open(INTERMEDIATE_PATH, "r"))
            if not isinstance(data, list):
                data = []
        except:
            data = []
    else:
        data = []

    data.append(entry)

    # keep last 1000
    if len(data) > 1000:
        data = data[-1000:]

    json.dump(data, open(INTERMEDIATE_PATH, "w"), indent=2)

    if needs_review:
        print(f"⚠ Logged for review → IP={ip} Action={action} Confidence={confidence:.2f}")


# =========================================================
# API Endpoints
# =========================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "scaler": os.path.exists(SCALER_PATH),
            "lstm": os.path.exists(LSTM_PATH),
            "xgb": os.path.exists(XGB_PATH)
        },
        "num_features": NUM_FEATURES,
        "human_review_enabled": True,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })


@app.route('/debug_info', methods=['GET'])
def debug_info():
    return jsonify({
        "scaler_exists": os.path.exists(SCALER_PATH),
        "lstm_exists": os.path.exists(LSTM_PATH),
        "xgb_exists": os.path.exists(XGB_PATH),
        "scaler_n_features": getattr(scaler, "n_features_in_", None),
        "num_features": NUM_FEATURES,
        "intermediate_path": INTERMEDIATE_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })


@app.route('/predict_seq', methods=['POST'])
def predict_seq():
    try:
        body = request.get_json()

        history = body.get("history")
        current = body.get("current")
        ip = body.get("ip","unknown")

        print(f"\n{'='*60}")
        print(f"🔍 NEW PREDICTION REQUEST - IP: {ip}")
        print(f"🔍 RAW REQUEST BODY: {body}")
        print(f"🔍 History: {history}")
        print(f"🔍 Current: {current}")
        print(f"🔍 Current length: {len(current) if current else 'None'}")
        print(f"{'='*60}")

        # Validate input - allow flexible feature count with padding
        if not history or len(history) < 1:
            return jsonify({"error":f"history must be list of feature vectors (min 1)"}),400

        if not current:
            return jsonify({"error":"current must be a feature vector"}),400

        # Convert to numpy
        hist_np = np.array(history, dtype=np.float32)
        cur_np = np.array(current, dtype=np.float32)
        
        # Auto-pad to 6 features if needed
        if cur_np.shape[0] < NUM_FEATURES:
            print(f"⚠ WARNING: Received {cur_np.shape[0]} features, padding to {NUM_FEATURES}")
            padding = np.zeros(NUM_FEATURES - cur_np.shape[0], dtype=np.float32)
            padding[-1] = 1.0  # Set always_on flag to 1.0
            cur_np = np.concatenate([cur_np, padding])
            
        if hist_np.shape[1] < NUM_FEATURES:
            print(f"⚠ WARNING: History has {hist_np.shape[1]} features, padding to {NUM_FEATURES}")
            padding = np.zeros((hist_np.shape[0], NUM_FEATURES - hist_np.shape[1]), dtype=np.float32)
            padding[:, -1] = 1.0  # Set always_on flag to 1.0
            hist_np = np.hstack([hist_np, padding])

        print(f"🔍 History shape: {hist_np.shape}")
        print(f"🔍 Current shape: {cur_np.shape}")
        print(f"🔍 Current features: {current}")
        print(f"   [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]")

        # --- BUILD FULL SEQUENCE: history + current ---
        full_sequence = np.vstack([hist_np, cur_np.reshape(1, -1)])
        
        print(f"🔍 Full sequence shape: {full_sequence.shape}")

        # --- LSTM suspicious score on FULL sequence ---
        susp_score = compute_suspicious_score(full_sequence)
        
        print(f"✅ Suspicious Score: {susp_score:.4f}")

        # --- Build XGB feature vector: current features + suspicious score ---
        # This creates 7 features: [pps, unique_ips, syn_ratio, rps, bandwidth, always_on, suspicious_score]
        feat_raw = np.concatenate([cur_np, np.array([susp_score], dtype=np.float32)])

        print(f"🔍 XGB feature vector (before scaling): {feat_raw}")

        # --- Scale ---
        if scaler is not None:
            feat_scaled = scaler.transform(feat_raw.reshape(1,-1))[0]
            print(f"🔍 XGB feature vector (after scaling): {feat_scaled}")
        else:
            feat_scaled = feat_raw
            print(f"⚠ No scaler loaded, using raw features")

        # --- XGBoost decision ---
        action, confidence = xgb_decision(feat_scaled.astype(np.float32))

        action_names = {0: "Allow", 1: "RateLimit", 2: "Block"}
        print(f"✅ Final Decision - Action: {action_names[action]}, Confidence: {confidence:.4f}")
        print(f"{'='*60}\n")

        # --- Log ---
        log_to_intermediate(history, current, ip, action, susp_score, confidence)

        return jsonify({
            "action": int(action),
            "action_name": action_names[action],
            "suspicious": float(susp_score),
            "confidence": float(confidence),
            "ip": ip,
            "needs_review": confidence < CONFIDENCE_THRESHOLD
        })

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/reload_models', methods=['POST'])
def reload_models():
    try:
        try_load()
        return jsonify({"reloaded":True,"timestamp":datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route('/intermediate_stats', methods=['GET'])
def intermediate_stats():
    if not os.path.exists(INTERMEDIATE_PATH):
        return jsonify({"total_predictions":0,"needs_review":0,"review_percentage":0.0})

    try:
        data=json.load(open(INTERMEDIATE_PATH,"r"))
        if not isinstance(data,list):
            data=[]
    except:
        data=[]

    total=len(data)
    needs=sum(1 for d in data if d.get("needs_review",False))
    pct = (needs/total*100) if total>0 else 0.0

    return jsonify({
        "total_predictions": total,
        "needs_review": needs,
        "review_percentage": round(pct,2)
    })


# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DDoS AUTONOMOUS FIREWALL ML SERVER (Flask)")
    print("=" * 80)
    print(f"Feature Set ({NUM_FEATURES} features):")
    print("  [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]")
    print("=" * 80)
    print("Human-in-the-Loop Continuous Learning Enabled")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Intermediate Log: {INTERMEDIATE_PATH}")
    print("=" * 80 + "\n")

    app.run(host="0.0.0.0", port=8000, debug=False)