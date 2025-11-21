# train_lstm.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# config
SEQ_LEN = 10
N_SAMPLES = 8000
BATCH = 64
EPOCHS = 12
OUT_PATH = "lstm.pt"

# synthetic generator
def gen_sequence(seq_len=SEQ_LEN):
    # base traffic
    base_pps = np.random.randint(10, 300)
    base_syn = np.random.random()*0.2
    base_ips = np.random.randint(1, 40)
    seq = []
    label = 0.0
    # decide if this sequence contains attack (prob 0.15)
    attack = np.random.random() < 0.15
    for t in range(seq_len):
        if attack and t > seq_len//3 and np.random.random() < 0.5:
            # attack spike
            pps = base_pps + np.random.randint(800, 2500)
            syn = min(1.0, base_syn + np.random.random()*0.8 + 0.2)
            unique = base_ips + np.random.randint(5, 40)
            label = max(label, 0.8)
        else:
            pps = max(1, int(base_pps + np.random.randint(-20, 40)))
            syn = min(1.0, base_syn + np.random.random()*0.15)
            unique = max(1, base_ips + np.random.randint(-2, 6))
        seq.append([pps, unique, syn])
    # for sequences with subtle anomalies, label between 0.4-0.7
    if not attack and np.random.random() < 0.05:
        label = 0.5
    # otherwise normal label ~ 0.05
    if label == 0.0:
        label = np.random.random()*0.1
    return np.array(seq, dtype=np.float32), float(label)

# dataset
class SeqDataset(Dataset):
    def __init__(self, n=N_SAMPLES):
        self.samples = [gen_sequence() for _ in range(n)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return x, np.array(y, dtype=np.float32)

# simple LSTM model like server
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

def train():
    ds = SeqDataset(N_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    model = SuspiciousLSTM()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # regression target [0,1]
    model.train()
    for epoch in range(EPOCHS):
        tot = 0.0
        for xb, yb in loader:
            xb = xb.float()
            yb = yb.float()
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} loss={tot/len(loader):.4f}")
    torch.save(model.state_dict(), OUT_PATH)
    print("Saved LSTM:", OUT_PATH)

if __name__ == "__main__":
    train()
