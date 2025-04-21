from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Model definition S
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

# --- FastAPI App ---
app = FastAPI()

# Add this to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can limit this to ["http://localhost:5173"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model ---
INPUT_FEATURES = 29  #  Change this based on final input size
MODEL_PATH = "rul_model_500.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedRULLSTM(input_size=INPUT_FEATURES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# --- Input format ---
class SequenceInput(BaseModel):
    sequence: list[list[float]]  # shape: [25][input_size]

@app.post("/predict")
def predict_rul(data: SequenceInput):
    try:
        seq = np.array(data.sequence, dtype=np.float32)
        print("Final layer weights:", model.fc[-1].weight)
        print("Sequence shape:", seq.shape)
        print("First row:", seq[0])


        if seq.shape != (25, INPUT_FEATURES):
            raise ValueError(f"Expected shape (25, {INPUT_FEATURES}), got {seq.shape}")

        with torch.no_grad():
            input_tensor = torch.tensor(seq).unsqueeze(0).to(device)  # shape: (1, 25, input_size)
            prediction = model(input_tensor).cpu().item()
            prediction = min(prediction, 150)
        print("Predicted RUL:", prediction)
        return {"rul": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

