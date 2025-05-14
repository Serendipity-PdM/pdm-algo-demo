import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Shift MLP Model ===
class ImprovedMLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load shift model
shift_scaler = joblib.load("mlp_shift_scaler.pkl")
shift_model = ImprovedMLP(input_size=11).to(device)
shift_model.load_state_dict(torch.load("mlp_shift_model.pth", map_location=device))
shift_model.eval()

class ShiftVectorRequest(BaseModel):
    vector: list[float]

@app.post("/predict_shift")
def predict_shift(data: ShiftVectorRequest):
    if len(data.vector) != 13:
        raise HTTPException(status_code=400, detail=f"Expected 11 input features, got {len(data.vector)}")

    input_array = np.array(data.vector).reshape(1, -1)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = shift_model(input_tensor).cpu().item()

    return {"predicted_time_cycles": float(pred)}

@app.get("/load_shift_data")
def load_shift_data():
    try:
        file_path = Path(__file__).parent / "datasets" / "shift-data" / "train_FD001_with_humans.csv"
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File load error: {e}")

@app.post("/append_shift_entry")
def append_shift_entry(req: ShiftVectorRequest):
    X = np.array(req.vector, dtype=float).reshape(1, -1)
    if X.shape[1] != 13:
        raise HTTPException(status_code=400, detail=f"Expected 13 input features, got {X.shape[1]}")

    X_scaled = shift_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_time_cycles = shift_model(X_tensor).cpu().item()

    risk = (
        "High" if predicted_time_cycles < 60
        else "Medium" if predicted_time_cycles < 120
        else "Low"
    )

    field_names = [
    "operator_id", "age", "avg_week_hours", "last_year_incidents",
    "shift_type_Afternoon", "shift_type_Night",
    "experience_Beginner", "experience_Intermediate", "experience_Experienced", "experience_Expert",
    "gender_Male"
]
    new_entry = {name: val for name, val in zip(field_names, req.vector)}
    new_entry["predicted_time_cycles"] = predicted_time_cycles
    new_entry["risk_factor"] = risk

    file_path = "pdm-backend/train_FD001_with_humans.csv"
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(file_path, index=False)

    return new_entry




# === MACHINE PREDICTION SECTION ===
WINDOW_SIZE = 25
RAW_FEATURES = ["os1", "os2", "os3"] + [f"s_{i}" for i in range(1, 22)]
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
FEATURE_NAME_TO_RAW_IDX = {name: idx for idx, name in enumerate(RAW_FEATURES)}
KEEP_RAW_INDICES = [FEATURE_NAME_TO_RAW_IDX[f] for f in RAW_FEATURES if f not in DROP_SENSORS]
SLOPE_SENSOR_NAMES = [f"s_{i}" for i in [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]]
SLOPE_KEPT_INDICES = [KEEP_RAW_INDICES.index(FEATURE_NAME_TO_RAW_IDX[s]) for s in SLOPE_SENSOR_NAMES]

# Paths for evaluation
TEST_FILE = Path("datasets/CMaps/test_FD001.txt")
RUL_FILE = Path("datasets/CMaps/RUL_FD001.txt")

# model definition for LSTM
class ImprovedRULLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.norm = torch.nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * self.num_directions, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        normed = self.norm(last_hidden)
        return self.fc(normed)

# load scaler & LSTM model
SCALER_PATH = Path("scaler_fd001.pkl")
MODEL_PATH = Path("rul_model_500.pth")
scaler = joblib.load(SCALER_PATH)
model = ImprovedRULLSTM(input_size=len(KEEP_RAW_INDICES) + len(SLOPE_KEPT_INDICES)).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

class SequenceRequest(BaseModel):
    sequence: list[list[float]]

@app.post("/predict")
def predict(req: SequenceRequest):
    seq = np.array(req.sequence, dtype=float)
    if seq.ndim != 2 or seq.shape != (WINDOW_SIZE, len(RAW_FEATURES)):
        raise HTTPException(400, f"sequence must be shape ({WINDOW_SIZE}, {len(RAW_FEATURES)})")
    reduced = seq[:, KEEP_RAW_INDICES]
    scaled = scaler.transform(reduced)
    x_time = np.arange(WINDOW_SIZE).reshape(-1, 1)
    slopes = []
    for idx in SLOPE_KEPT_INDICES:
        y = scaled[:, idx].reshape(-1, 1)
        slopes.append(LinearRegression().fit(x_time, y).coef_[0][0])
    extended = np.hstack((scaled, np.tile(slopes, (WINDOW_SIZE, 1))))
    tensor = torch.tensor(extended, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).cpu().item()
    return {"rul": float(min(pred, 150.0))}

# --- Start Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
