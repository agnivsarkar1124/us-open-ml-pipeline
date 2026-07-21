from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

app = FastAPI(title="US Open Prediction API")

# Enable CORS so your GitHub Pages site can make API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from agnivsarkar1124.github.io
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request payload schema
class MatchupRequest(BaseModel):
    pA_name: str
    pB_name: str
    elo_diff: float
    h2h_diff: float

# Dummy model initialized for demo (In practice, load your saved joblib/pickle model)
X_dummy = np.random.randn(100, 2)
y_dummy = np.random.randint(0, 2, 100)
model = HistGradientBoostingClassifier().fit(X_dummy, y_dummy)

@app.get("/")
def health_check():
    return {"status": "online", "model": "HistGradientBoosting"}

@app.post("/predict")
def predict_matchup(req: MatchupRequest):
    feats = pd.DataFrame([[req.elo_diff, req.h2h_diff]], columns=["elo_diff", "h2h_diff"])
    prob = float(model.predict_proba(feats)[0][1])
    
    winner = req.pA_name if prob >= 0.5 else req.pB_name
    confidence = prob if prob >= 0.5 else (1 - prob)
    
    return {
        "winner": winner,
        "confidence": round(confidence * 100, 1),
        "raw_prob_pA": round(prob, 3)
    }
