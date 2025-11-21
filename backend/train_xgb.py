# train_xgb.py
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import torch
import torch.nn as nn

# paths
LSTM_PATH = "lstm.pt"
SCALER_PATH = "scaler.joblib"
XGB_PATH = "xgb.joblib"

# load lstm model class
class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden,1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.act(self.fc(last)).squeeze(1)

def compute_suspicious_with_lstm(lstm_model, history):
    if lstm_model is None:
        # fallback heuristic
        pps = history[-1,0]; syn = history[-1,2]
        s = min(1.0, (pps/2000)*0.6 + syn*0.6)
        return s
    arr = history.astype("float32")[None,...]
    with torch.no_grad():
        t = torch.from_numpy(arr)
        out = lstm_model(t).numpy()
        return float(out[0])

# generate labeled dataset (three classes)
def gen_snapshot(lstm_model=None):
    # generate a short sequence
    seq_len=10
    base_pps = np.random.randint(10, 300)
    base_syn = np.random.random()*0.2
    base_ips = np.random.randint(1, 40)
    seq = []
    attack=False
    for t in range(seq_len):
        if np.random.random() < 0.06 and t>3:
            attack=True
            pps = base_pps + np.random.randint(800,2500)
            syn = min(1.0, base_syn + np.random.random()*0.8 + 0.2)
            uniq = base_ips + np.random.randint(5,50)
        else:
            pps = max(1, int(base_pps + np.random.randint(-20,40)))
            syn = min(1.0, base_syn + np.random.random()*0.15)
            uniq = max(1, base_ips + np.random.randint(-2,6))
        seq.append([pps, uniq, syn])
    hist = np.array(seq)
    suspicious = compute_suspicious_with_lstm(lstm_model, hist)
    current = hist[-1]
    # label heuristic (we'll convert to classes)
    pps_val = current[0]
    if pps_val > 1500 or suspicious>0.8 or current[2]>0.7:
        label = 2
    elif pps_val > 800 or suspicious>0.5:
        label = 1
    else:
        label = 0
    # features: pps, uniq, syn, suspicious
    feat = np.concatenate([current, np.array([suspicious])])
    return feat, label

def main(n=8000):
    # load lstm if exists
    lstm = None
    if os.path.exists(LSTM_PATH):
        lstm = SuspiciousLSTM()
        lstm.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
        lstm.eval()
        print("Loaded LSTM for suspicious scoring")
    else:
        print("No LSTM found; using heuristic for suspicious score")

    X=[]
    y=[]
    for i in range(n):
        feat, lab = gen_snapshot(lstm)
        X.append(feat)
        y.append(lab)
    X = np.array(X)
    y = np.array(y)
    # scale features (we will save scaler)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print("Saved scaler:", SCALER_PATH)
    # train xgboost classifier
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.18, random_state=42, stratify=y)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="mlogloss")
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    print("XGBoost classification report:\n", classification_report(yte, preds))
    joblib.dump(clf, XGB_PATH)
    print("Saved xgb:", XGB_PATH)

if __name__ == "__main__":
    main(8000)
