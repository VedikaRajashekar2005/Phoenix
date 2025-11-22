# Phoenix

Phoenix is a project focused on DDoS detection, continuous learning, and human review workflows. It includes backend machine learning models, frontend web applications, and datasets for training and prediction.

## Folder Structure

- `dataset/`  
  Contains datasets for training, review history, and synthetic traffic data.
  - `ddos_central_dataset.json` — Main dataset for DDoS detection.
  - `ddos_intermediate.json` — Intermediate processed dataset.
  - `ddos_review_history.json` — Human review history.
  - `synthetic_traffic.csv` — Synthetic network traffic data.

- `htil/`  
  Human-in-the-loop (HTIL) system for DDoS detection.
  - `backend/` — Python scripts for continuous learning and human review.
    - `ContinuousLearning_DDoS.py` — Continuous learning pipeline.
    - `HumanReview_DDoS.py` — Human review workflow.
    - `train_lstm.py`, `train_xgb.py` — Model training scripts (LSTM, XGBoost).
    - `requirements.txt` — Python dependencies.
  - `frontend/` — React web app for HTIL interface.
    - `public/`, `src/` — Standard React structure.

- `models/`  
  Pre-trained models and scalers.
  - `lstm.pt` — LSTM model (PyTorch).
  - `model.pt` — Additional model file.
  - `scaler.joblib` — Scaler for preprocessing.
  - `xgb.joblib` — XGBoost model.

- `prediction/`  
  Prediction service for DDoS detection.
  - `backend/` — Python server for ML predictions (`ServerML_DDoS.py`).
  - `frontend/` — Vite/React web app for prediction interface.
    - `public/`, `src/` — Standard Vite/React structure.
    - `components/FirewallDemo.jsx` — Demo component for firewall simulation.

## Getting Started

### Backend Setup
1. Navigate to the backend folder (e.g., `htil/backend` or `prediction/backend`).
2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run backend scripts as needed:
   ```powershell
   python ContinuousLearning_DDoS.py
   python HumanReview_DDoS.py
   python ServerML_DDoS.py
   ```

### Frontend Setup
1. Navigate to the frontend folder (e.g., `htil/frontend` or `prediction/frontend`).
2. Install Node.js dependencies:
   ```powershell
   npm install
   ```
3. Start the development server:
   ```powershell
   npm start
   # or for Vite projects
   npm run dev
   ```

## Usage
- Train models using scripts in `htil/backend`.
- Use the HTIL frontend for human review and continuous learning.
- Deploy prediction backend and frontend for real-time DDoS detection.


