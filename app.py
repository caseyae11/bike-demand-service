import json
import os
import joblib
import numpy as np
import torch
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bike Demand Prediction Service", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

with open(os.path.join(ARTIFACTS_DIR, "feature_cols.json"), "r") as f:
    FEATURE_COLS = json.load(f)

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

# Model matches the training architecture (plain Sequential)
model = nn.Sequential(
    nn.Linear(len(FEATURE_COLS), 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

state = torch.load(os.path.join(ARTIFACTS_DIR, "bike_mlp.pt"), map_location="cpu")
model.load_state_dict(state)
model.eval()

class PredictRequest(BaseModel):
    season: float
    yr: float
    mnth: float
    hr: float
    holiday: float
    weekday: float
    workingday: float
    weathersit: float
    temp: float
    atemp: float
    hum: float
    windspeed: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    raw = req.model_dump()
    x = np.array([[raw[c] for c in FEATURE_COLS]], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)

    with torch.no_grad():
        pred = model(torch.tensor(x_scaled)).item()

    return {"prediction_cnt": float(max(0.0, pred))}
