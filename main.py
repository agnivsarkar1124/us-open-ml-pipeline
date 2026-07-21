from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

app = FastAPI(title="US Open Prediction API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached data and model
all_matches = None
h2h_tracker = {}
model = None
player_list = []

def train_model():
    global all_matches, h2h_tracker, model, player_list
    base_url = "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{}.csv"
    dfs = []
    
    # Fetch historical ATP data
    for year in [2020, 2021, 2022, 2023]:
        try:
            df = pd.read_csv(base_url.format(year))
            df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
            dfs.append(df)
        except Exception:
            pass
            
    if not dfs:
        raise RuntimeError("Failed to load match datasets.")
        
    all_matches = pd.concat(dfs, ignore_index=True).sort_values("tourney_date")

    # Extract dynamic list of unique players
    winners = all_matches["winner_name"].dropna().unique()
    losers = all_matches["loser_name"].dropna().unique()
    player_list = sorted(list(set(winners).union(set(losers))))

    # Feature engineering
    us_open_matches = all_matches[
        all_matches["tourney_name"].str.contains("US Open", case=False, na=False)
    ].copy()
    
    ml_rows = []
    np.random.seed(42)

    for _, row in us_open_matches.iterrows():
        winner, loser = row["winner_name"], row["loser_name"]
        swap = np.random.rand() > 0.5
        pA, pB = (loser, winner) if swap else (winner, loser)
        target = 0 if swap else 1

        pair_key = tuple(sorted([pA, pB]))
        h2h_data = h2h_tracker.get(pair_key, {pA: 0, pB: 0})
        h2h_diff = h2h_data.get(pA, 0) - h2h_data.get(pB, 0)

        if pair_key not in h2h_tracker:
            h2h_tracker[pair_key] = {winner: 1, loser: 0}
        else:
            h2h_tracker[pair_key][winner] = h2h_tracker[pair_key].get(winner, 0) + 1

        ml_rows.append({"elo_diff": 0, "h2h_diff": h2h_diff, "target": target})

    ml_df = pd.DataFrame(ml_rows)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(ml_df[["elo_diff", "h2h_diff"]], ml_df["target"])

# Train model when FastAPI starts up
@app.on_event("startup")
def startup_event():
    train_model()

# Request payload schema from index.html
class MatchupRequest(BaseModel):
    pA_name: str
    pB_name: str

@app.get("/")
def health_check():
    return {"status": "online", "players_loaded": len(player_list)}

@app.get("/players")
def get_players():
    """Returns dynamic list of players fetched from Kadantte ATP dataset."""
    if not player_list:
        raise HTTPException(status_code=500, detail="Player list not initialized")
    return {"players": player_list}

@app.post("/predict")
def predict_matchup(req: MatchupRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    p1_name = req.pA_name
    p2_name = req.pB_name

    # Calculate H2H feature dynamically
    pair_key = tuple(sorted([p1_name, p2_name]))
    h2h_data = h2h_tracker.get(pair_key, {p1_name: 0, p2_name: 0})
    h2h_diff = h2h_data.get(p1_name, 0) - h2h_data.get(p2_name, 0)

    feats = pd.DataFrame([[0, h2h_diff]], columns=["elo_diff", "h2h_diff"])
    prob = float(model.predict_proba(feats)[0][1])

    winner = p1_name if prob >= 0.5 else p2_name
    confidence = prob if prob >= 0.5 else (1 - prob)

    return {
        "winner": winner,
        "confidence": round(confidence * 100, 1),
        "elo_diff": 0,
        "h2h_diff": h2h_diff
    }
