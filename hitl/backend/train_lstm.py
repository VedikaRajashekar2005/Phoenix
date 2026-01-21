import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random, os, json
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============ CONFIG ============
SEQ_LEN = 12          # sequence length for temporal patterns
N_SAMPLES = 15000     # larger dataset for better learning
BATCH = 64
EPOCHS = 20           # more epochs for better convergence
CONTINUE_EPOCHS = 8   # when continuing from existing model
OUT_PATH = os.getenv("LSTM_PATH")
CENTRAL_DATA_PATH = os.getenv("CENTRAL_DATA_PATH")

# ============================================================
# Feature definition per timestep (6 features):
# [pps, unique_ips, syn_ratio, rps, bandwidth, always_on_flag]
# ============================================================

def gen_sequence(seq_len=SEQ_LEN):
    base_pps = np.random.randint(50, 400)
    base_syn = np.random.random() * 0.15
    base_ips = np.random.randint(5, 50)
    base_rps = np.random.randint(20, 200)
    base_bw  = np.random.randint(1, 10)  # Mbps baseline

    seq = []
    suspicious_label = 0.0

    # INCREASED attack probability to create more variety
    attack_type = np.random.choice(
        ["normal","protocol","application","volumetric"],
        p=[0.50, 0.20, 0.15, 0.15]  # 50% attack scenarios
    )

    # Attack intensity varies
    intensity = np.random.random() * 0.6 + 0.3  # 0.3 to 0.9

    for t in range(seq_len):

        # default normal behavior
        pps = base_pps + np.random.randint(-20, 30)
        syn = min(1.0, base_syn + np.random.random()*0.05)
        uniq = base_ips + np.random.randint(-3, 5)
        rps = base_rps + np.random.randint(-10, 20)
        bw  = base_bw + np.random.randint(-1, 2)

        # ===== PROTOCOL ATTACK (SYN flood / PPS spike) =====
        if attack_type == "protocol" and t > seq_len//3:
            if np.random.random() < 0.8:  # 80% chance during attack window
                # More aggressive - reaches higher values
                pps = base_pps + np.random.randint(1500, 5000) * intensity
                syn = min(1.0, base_syn + np.random.random() * 0.9 * intensity)
                uniq = base_ips + np.random.randint(20, 100) * intensity
                
                # Suspicious score scales with intensity
                suspicious_label = max(suspicious_label, 0.7 + intensity * 0.2)

        # ===== APPLICATION-LAYER ATTACK (HTTP RPS flood) =====
        if attack_type == "application" and t > seq_len//3:
            if np.random.random() < 0.7:  # 70% chance
                rps = base_rps + np.random.randint(800, 3000) * intensity
                uniq = base_ips + np.random.randint(10, 50) * intensity
                pps = base_pps + np.random.randint(200, 800) * intensity
                
                suspicious_label = max(suspicious_label, 0.65 + intensity * 0.25)

        # ===== VOLUMETRIC BANDWIDTH ATTACK =====
        if attack_type == "volumetric" and t > seq_len//3:
            if np.random.random() < 0.7:  # 70% chance
                bw = base_bw + np.random.randint(200, 800) * intensity
                pps = base_pps + np.random.randint(800, 2500) * intensity
                uniq = base_ips + np.random.randint(15, 60) * intensity
                
                suspicious_label = max(suspicious_label, 0.70 + intensity * 0.2)

        # ===== Always-on monitoring flag (Cloudflare concept) =====
        always_on = 1.0  

        # Ensure values stay reasonable (no negative)
        pps = max(1, int(pps))
        uniq = max(1, int(uniq))
        rps = max(1, int(rps))
        bw = max(1, int(bw))
        syn = max(0.0, min(1.0, syn))

        seq.append([pps, uniq, syn, rps, bw, always_on])

    # Normal traffic gets low score with some variation
    if suspicious_label == 0.0:
        suspicious_label = np.random.random() * 0.20  # 0-0.20 for normal

    # Add some noise to make it more realistic
    suspicious_label = np.clip(suspicious_label + np.random.normal(0, 0.05), 0, 1)

    return np.array(seq, dtype=np.float32), float(suspicious_label)


# ============================================================
# Dataset
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, samples=None, n=N_SAMPLES):
        if samples:
            self.samples = [
                (np.array(s['history'],dtype=np.float32),
                 np.array(s.get('suspicious_score',
                        s.get('predicted_suspicious',0.0)),dtype=np.float32))
                for s in samples
            ]
        else:
            print(f"🔄 Generating {n} synthetic sequences...")
            self.samples = [gen_sequence() for _ in range(n)]
            
            # Verify we have good distribution of suspicious scores
            scores = [s[1] for s in self.samples]
            print(f"📊 Suspicious Score Distribution:")
            print(f"   Low (0-0.3):    {sum(1 for s in scores if s < 0.3):5d} ({sum(1 for s in scores if s < 0.3)/n*100:5.1f}%)")
            print(f"   Medium (0.3-0.6): {sum(1 for s in scores if 0.3 <= s < 0.6):5d} ({sum(1 for s in scores if 0.3 <= s < 0.6)/n*100:5.1f}%)")
            print(f"   High (0.6-1.0):  {sum(1 for s in scores if s >= 0.6):5d} ({sum(1 for s in scores if s >= 0.6)/n*100:5.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return x, np.array(float(y),dtype=np.float32)


# ============================================================
# LSTM Model (matches server)
# ============================================================

class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden,1)
        self.act = nn.Sigmoid()

    def forward(self,x):
        out,_ = self.lstm(x)
        last = out[:,-1,:]
        return self.act(self.fc(last)).squeeze(1)


# ============================================================
# Training
# ============================================================

def train():
    samples=None
    if os.path.exists(CENTRAL_DATA_PATH):
        try:
            data=json.load(open(CENTRAL_DATA_PATH))
            if data and len(data) > 0:
                samples=data
                print(f"✓ Loaded {len(samples)} real samples from central dataset")
        except Exception as e:
            print(f"⚠️  Could not load central data: {e}")

    if samples and len(samples) > 500:
        ds=SeqDataset(samples=samples)
        epochs=CONTINUE_EPOCHS
        print("📚 Mode: Continuous Learning")
    else:
        ds=SeqDataset(n=N_SAMPLES)
        epochs=EPOCHS
        print("📚 Mode: Initial Training")

    # ============================================================
    # Save newly generated synthetic samples into central dataset
    # ============================================================

    if samples is None or len(samples) <= 500:   # means synthetic data was generated
        os.makedirs(os.path.dirname(CENTRAL_DATA_PATH), exist_ok=True)

        central_data = []
        if os.path.exists(CENTRAL_DATA_PATH):
            try:
                central_data = json.load(open(CENTRAL_DATA_PATH))
                if not isinstance(central_data, list):
                    central_data = []
            except:
                central_data = []

        # convert generated dataset to json format
        print(f"💾 Storing {len(ds.samples)} samples into central dataset...")
        for seq, score in ds.samples:
            central_data.append({
                "history": seq.tolist(),
                "suspicious_score": float(score)
            })

        # Keep last 20k samples to avoid file getting too large
        if len(central_data) > 20000:
            central_data = central_data[-20000:]

        json.dump(central_data, open(CENTRAL_DATA_PATH,"w"), indent=2)
        print(f"✓ Saved {len(central_data)} total samples to central dataset")

    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,drop_last=True)

    model=SuspiciousLSTM()

    if os.path.exists(OUT_PATH):
        try:
            model.load_state_dict(torch.load(OUT_PATH,map_location="cpu"))
            print("✓ Loaded existing LSTM – continuing training")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}")
            print("🆕 Starting fresh model")

    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    criterion=nn.MSELoss()

    model.train()
    print(f"\n🚀 Training LSTM for {epochs} epochs...")
    print("="*60)
    
    for e in range(epochs):
        total_loss=0
        for xb,yb in loader:
            xb=xb.float()
            yb=yb.float()
            pred=model(xb)
            loss=criterion(pred,yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        
        avg_loss = total_loss/len(loader)
        print(f"Epoch {e+1:2d}/{epochs}  Loss={avg_loss:.4f}")

    print("="*60)
    
    # Save model
    torch.save(model.state_dict(),OUT_PATH)
    print(f"✓ Saved LSTM → {OUT_PATH}")
    
    # Quick validation check
    print("\n📊 Validation Check (5 random samples):")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            idx = np.random.randint(0, len(ds))
            x, y_true = ds[idx]
            x_batch = torch.from_numpy(x[None, ...]).float()
            y_pred = model(x_batch).item()
            print(f"   Sample {i+1}: True={y_true:.3f}, Predicted={y_pred:.3f}, Error={abs(y_true-y_pred):.3f}")
    
    print("\n✅ Training complete!")


if __name__=="__main__":
    train()