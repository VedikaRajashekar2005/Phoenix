from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import torch
import joblib
from agent import DQN

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load RL Model
model_dict = joblib.load("trained_firewall_model.pkl")
model = DQN()
model.load_state_dict(model_dict)
model.eval()

def predict_action(features):
    x = torch.FloatTensor(features)
    with torch.no_grad():
        q = model(x)
    return int(torch.argmax(q).item())

@socketio.on("packet")
def handle_packet(data):
    features = np.array(data["features"], dtype=np.float32)
    action = predict_action(features)
    emit("firewall_decision", {"action": action, "ip": data.get("ip")})

@app.route("/")
def home():
    return "<h2>RL Firewall Flask Server Running</h2>"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)
