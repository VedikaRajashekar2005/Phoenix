import numpy as np, os, json, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

LSTM_PATH = os.getenv("LSTM_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
XGB_PATH = os.getenv("XGB_PATH")
CENTRAL_DATA_PATH = os.getenv("CENTRAL_DATA_PATH")


# ============================================================
# LSTM definition (must match train_lstm.py)
# ============================================================

class SuspiciousLSTM(nn.Module):
    def __init__(self,input_dim=6,hidden=64,num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden,num_layers,
                            batch_first=True,dropout=0.3)
        self.fc = nn.Linear(hidden,1)
        self.act = nn.Sigmoid()
    def forward(self,x):
        o,_ = self.lstm(x)
        last=o[:,-1,:]
        return self.act(self.fc(last)).squeeze(1)


def compute_suspicious(model,history):
    arr = history.astype("float32")[None,...]
    with torch.no_grad():
        t=torch.from_numpy(arr)
        return float(model(t).numpy()[0])


# ============================================================
# Synthetic snapshot generator with BALANCED 3-class labels
# ============================================================

def gen_snapshot(lstm):
    seq,_ = gen_sequence()  # reuse same generator from LSTM logic
    suspicious = compute_suspicious(lstm,seq)

    # Current timestep features
    cur = seq[-1]

    pps, uniq, syn, rps, bw, always_on = cur

    # IMPROVED DECISION LOGIC - creates balanced classes
    # Strategy: Use thresholds that create ~33% each class
    
    # High severity → BLOCK (class 2)
    if (pps > 2000 or bw > 400 or rps > 1500) or (suspicious > 0.8):
        label = 2  # Block
    
    # Medium severity → RATE LIMIT (class 1)
    elif (pps > 500 or bw > 50 or rps > 300) or (suspicious > 0.4):
        label = 1  # Rate limit
    
    # Low severity → ALLOW (class 0)
    else:
        label = 0  # Allow

    # Add some randomness to create edge cases
    if np.random.random() < 0.05:  # 5% random flips
        label = np.random.choice([0, 1, 2])

    # final feature vector
    feat = np.concatenate([cur, [suspicious]])
    return feat, label


# ============================================================
# Bring gen_sequence from LSTM code - MORE ATTACK VARIETY
# ============================================================

def gen_sequence(seq_len=12):
    base_pps = np.random.randint(50, 400)
    base_syn = np.random.random()*0.15
    base_ips = np.random.randint(5,50)
    base_rps = np.random.randint(20,200)
    base_bw  = np.random.randint(1,10)

    seq=[]
    
    # INCREASED attack probability to get more variety
    attack_type=np.random.choice(
        ["normal","protocol","application","volumetric"],
        p=[0.50, 0.20, 0.15, 0.15]  # More attacks!
    )
    
    for t in range(seq_len):
        pps = base_pps + np.random.randint(-20,30)
        syn = min(1.0,base_syn+np.random.random()*0.05)
        uniq= base_ips + np.random.randint(-3,5)
        rps = base_rps + np.random.randint(-10,20)
        bw  = base_bw + np.random.randint(-1,2)
        always_on=1.0

        # PROTOCOL ATTACK - more aggressive
        if attack_type=="protocol" and t>seq_len//3 and np.random.random()<0.8:
            pps = base_pps + np.random.randint(1500, 5000)  # Higher PPS
            syn = min(1.0,base_syn+np.random.random()*0.9)
            uniq=base_ips+np.random.randint(20,100)

        # APPLICATION ATTACK - more aggressive
        if attack_type=="application" and t>seq_len//3 and np.random.random()<0.7:
            rps=base_rps+np.random.randint(800, 3000)  # Higher RPS
            uniq=base_ips+np.random.randint(10,50)

        # VOLUMETRIC ATTACK - more aggressive
        if attack_type=="volumetric" and t>seq_len//3 and np.random.random()<0.7:
            bw=base_bw+np.random.randint(200, 800)  # Higher bandwidth
            pps=base_pps+np.random.randint(800, 2500)

        seq.append([pps,uniq,syn,rps,bw,always_on])

    return np.array(seq,dtype=np.float32),0

# ============================================================
# Load LSTM
# ============================================================

def load_lstm():
    model=SuspiciousLSTM()
    if os.path.exists(LSTM_PATH):
        model.load_state_dict(torch.load(LSTM_PATH,map_location="cpu"))
        model.eval()
        print("✓ Loaded LSTM")
    else:
        print("✗ No LSTM found – run train_lstm.py first")
    return model


# ============================================================
# Main XGBoost Training with BALANCED DATA
# ============================================================

def main(n=15000):  # Increased sample size

    lstm=load_lstm()

    # Try real dataset
    X_real,y_real=None,None
    if os.path.exists(CENTRAL_DATA_PATH):
        try:
            data=json.load(open(CENTRAL_DATA_PATH))
            X_real=[]
            y_real=[]
            for s in data:
                hist=np.array(s["history"],dtype=np.float32)
                suspicious=compute_suspicious(lstm,hist)
                cur=hist[-1]
                feat=np.concatenate([cur,[suspicious]])
                X_real.append(feat)
                y_real.append(s.get("action",0))
            X_real=np.array(X_real)
            y_real=np.array(y_real)
            print(f"✓ Loaded {len(X_real)} real samples")
        except: pass

    if X_real is not None and len(X_real)>200 and len(np.unique(y_real)) >= 3:
        X = X_real
        y = y_real
        print("Mode: Continuous learning")
    else:
        print("Mode: Synthetic balanced dataset (real data insufficient or single-class)")
        X=[]; y=[]
        for _ in range(n):
            f,l = gen_snapshot(lstm)
            X.append(f); y.append(l)
        X=np.array(X); y=np.array(y)


    # ============================================================
    # CHECK CLASS BALANCE - CRITICAL!
    # ============================================================
    
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\n📊 Class Distribution:")
    print(f"   Class 0 (Allow):     {class_dist.get(0, 0):5d} ({class_dist.get(0, 0)/len(y)*100:5.1f}%)")
    print(f"   Class 1 (RateLimit): {class_dist.get(1, 0):5d} ({class_dist.get(1, 0)/len(y)*100:5.1f}%)")
    print(f"   Class 2 (Block):     {class_dist.get(2, 0):5d} ({class_dist.get(2, 0)/len(y)*100:5.1f}%)")
    
    # Ensure all 3 classes exist
    while len(np.unique(y)) < 3:
        f,l = gen_snapshot(lstm)
        X = np.vstack([X, f])
        y = np.append(y, l)

    # ============================================================
    # Save newly generated synthetic samples into central dataset
    # ============================================================

    if X_real is None or len(X_real) <= 200:   # synthetic mode
        os.makedirs(os.path.dirname(CENTRAL_DATA_PATH), exist_ok=True)

        central_data = []
        if os.path.exists(CENTRAL_DATA_PATH):
            try:
                central_data = json.load(open(CENTRAL_DATA_PATH))
            except:
                central_data = []

        # Store samples
        for i in range(len(X)):
            seq, _ = gen_sequence()  # full 12×6 sequence            
            central_data.append({
                "history": seq.tolist(),
                "action": int(y[i])
            })

        json.dump(central_data, open(CENTRAL_DATA_PATH,"w"), indent=2)
        print(f"✓ Stored {len(X)} action samples into central dataset")

    # ============================================================
    # TRAIN WITH CLASS WEIGHTS for imbalanced data
    # ============================================================
    
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    joblib.dump(scaler,SCALER_PATH)
    print(f"✓ Saved scaler")

    # Calculate class weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y)

    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,
                                      random_state=42,stratify=y)
    wtr,wte = train_test_split(sample_weights,test_size=0.2,
                                random_state=42,stratify=y)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,                  # EXPLICITLY set to 3 classes
        eval_metric="mlogloss",
        random_state=42
    )

    print(f"\n🚀 Training XGBoost with {len(Xtr)} samples...")
    clf.fit(Xtr, ytr, sample_weight=wtr)

    preds=clf.predict(Xte)
    print("\n📊 Classification Report:")

    labels_present = sorted(list(set(yte)))
    target_map = {0:"Allow", 1:"RateLimit", 2:"Block"}
    target_names = [target_map[l] for l in labels_present]

    print(classification_report(
        yte,
        preds,
        labels=labels_present,
        target_names=target_names
    ))

    joblib.dump(clf,XGB_PATH)
    print(f"✓ Saved XGBoost → {XGB_PATH}")
    print(f"\n✅ Training complete! Model supports all 3 classes.")


if __name__=="__main__":
    main()