import numpy as np
import torch
import torch.optim as optim
import joblib
from agent import DQN

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

def generate_sample():
    """Simulated traffic features"""
    pps = np.random.randint(1, 2000)
    unique_ips = np.random.randint(1, 50)
    syn_ratio = np.random.random()
    suspicious = 0.4 * (pps / 2000) + 0.6 * syn_ratio
    
    features = np.array([pps, unique_ips, syn_ratio, suspicious], dtype=np.float32)

    # rule-based labels
    if pps > 1500 or suspicious > 0.7:
        label = 2  # block
    elif pps > 800:
        label = 1  # rate-limit
    else:
        label = 0  # allow

    return features, label


def train_model(epochs=4000):
    for epoch in range(epochs):
        features, label = generate_sample()
        x = torch.FloatTensor(features)
        y = torch.LongTensor([label])

        pred = model(x)
        loss = loss_fn(pred.unsqueeze(0), torch.nn.functional.one_hot(y, 3).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")

    joblib.dump(model.state_dict(), "trained_firewall_model.pkl")
    print("Model saved.")

if __name__ == "__main__":
    train_model()
