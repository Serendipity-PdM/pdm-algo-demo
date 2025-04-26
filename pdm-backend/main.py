from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "rul_model_500.pth"
SCALER_PATH = ROOT / "scaler_fd001.pkl"
WINDOW_SIZE = 25
RAW_FEATURES = 24
IN_FEATURES = 29

SELECTED = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23]
SLOPE_IDX = [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]


class ImprovedRULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        normed = self.norm(last_hidden)
        return self.fc(normed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FEATURES = 29
model = ImprovedRULLSTM(input_size=INPUT_FEATURES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

SCALER = joblib.load(SCALER_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceInput(BaseModel):
    sequence: list[list[float]]


@app.post("/predict")
def predict_rul(data: SequenceInput):
    try:
        seq = np.asarray(data.sequence, dtype=np.float32)
        if seq.shape != (WINDOW_SIZE, RAW_FEATURES):
            raise ValueError("Expected shape (25, 24) for 'sequence'.")

        # 1. drop unused columns / reorder exactly as during training
        seq = seq[:, SELECTED]                               # (25, 17)

        # 2. standardise with the **trained** scaler
        # same stats as training
        seq = SCALER.transform(seq)

        # 3. fit linear slopes on selected sensors (using scaled values)
        t = np.arange(WINDOW_SIZE).reshape(-1, 1)
        slopes = [
            LinearRegression().fit(t, seq[:, i].reshape(-1, 1)).coef_[0][0]
            for i in SLOPE_IDX
        ]
        seq_full = np.hstack(
            (seq, np.tile(slopes, (WINDOW_SIZE, 1))))  # (25, 29)

        # 4. predict
        with torch.no_grad():
            x = torch.tensor(seq_full).unsqueeze(
                0).to(DEVICE)          # (1,25,29)
            rul = model(x).cpu().item()

        return {"rul": round(float(rul), 2)}

    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
