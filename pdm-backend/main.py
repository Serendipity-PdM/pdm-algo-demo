import os
import sys
import torch
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# configuration
WINDOW_SIZE = 25
RAW_FEATURES = ["os1", "os2", "os3"] + [f"s_{i}" for i in range(1, 22)]
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
FEATURE_NAME_TO_RAW_IDX = {name: idx for idx, name in enumerate(RAW_FEATURES)}
KEEP_RAW_INDICES = [FEATURE_NAME_TO_RAW_IDX[f]
                    for f in RAW_FEATURES if f not in DROP_SENSORS]
SLOPE_SENSOR_NAMES = [f"s_{i}" for i in [
    2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]]
SLOPE_KEPT_INDICES = [KEEP_RAW_INDICES.index(
    FEATURE_NAME_TO_RAW_IDX[s]) for s in SLOPE_SENSOR_NAMES]

# Paths for evaluation
TEST_FILE = Path("datasets/CMaps/test_FD001.txt")
RUL_FILE = Path("datasets/CMaps/RUL_FD001.txt")

# model definition


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


# load scaler & model
SCALER_PATH = Path("scaler_fd001.pkl")
MODEL_PATH = Path("rul_model_500.pth")
scaler = joblib.load(SCALER_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedRULLSTM(input_size=len(
    KEEP_RAW_INDICES) + len(SLOPE_KEPT_INDICES)).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceRequest(BaseModel):
    sequence: list[list[float]]


@app.post("/predict")
def predict(req: SequenceRequest):
    seq = np.array(req.sequence, dtype=float)
    if seq.ndim != 2 or seq.shape != (WINDOW_SIZE, len(RAW_FEATURES)):
        raise HTTPException(400, f"sequence must be shape ({
                            WINDOW_SIZE}, {len(RAW_FEATURES)})")
    # preprocess
    reduced = seq[:, KEEP_RAW_INDICES]
    scaled = scaler.transform(reduced)
    x_time = np.arange(WINDOW_SIZE).reshape(-1, 1)
    slopes = []
    for idx in SLOPE_KEPT_INDICES:
        y = scaled[:, idx].reshape(-1, 1)
        slopes.append(LinearRegression().fit(x_time, y).coef_[0][0])
    extended = np.hstack((scaled, np.tile(slopes, (WINDOW_SIZE, 1))))
    tensor = torch.tensor(
        extended, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).cpu().item()
    return {"rul": float(min(pred, 150.0))}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
