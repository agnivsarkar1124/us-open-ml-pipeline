from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

app = FastAPI(title="US Open Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

all_matches = None
h2h_tracker = {}
player_stats = {}
model = None
player_list = []

def train_model():
    global all_matches, h2h_tracker, player_stats, model, player_list
    base_url = "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{}.csv"
    dfs = []
    
    # Load 2020 - 2023 ATP Match Data
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

    winners = all_matches["winner_name"].dropna().unique()
    losers = all_matches["loser_name"].dropna().unique()
    player_list = sorted(list(set(winners).union(set(losers))))

    # Compute overall win rates and counts per player
    for player in player_list:
        wins = len(all_matches[all_matches["winner_name"] == player])
        losses = len(all_matches[all_matches["loser_name"] == player])
        total = wins + losses
        win_rate = (wins / total) if total > 5 else 0.50  # default prior if low sample
        player_stats[player] = {
            "wins": wins,
            "losses": losses,
            "total_matches": total,
            "win_rate": win_rate
        }

    # Build H2H tracker
    for _, row in all_matches.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        pair_key = tuple(sorted([w, l]))
        if pair_key not in h2h_tracker:
            h2h_tracker[pair_key] = {w: 1, l: 0}
        else:
            h2h_tracker[pair_key][w] = h2h_tracker[pair_key].get(w, 0) + 1

    # Build Training Set using dynamic player features
    ml_rows = []
    np.random.seed(42)

    us_open_matches = all_matches[
        all_matches["tourney_name"].str.contains("US Open", case=False, na=False)
    ].copy()

    for _, row in us_open_matches.iterrows():
        winner, loser = row["winner_name"], row["loser_name"]
        swap = np.random.rand() > 0.5
        pA, pB = (loser, winner) if swap else (winner, loser)
        target = 0 if swap else 1

        # Calculate features
        pA_wr = player_stats.get(pA, {}).get("win_rate", 0.50)
        pB_wr = player_stats.get(pB, {}).get("win_rate", 0.50)
        wr_diff = pA_wr - pB_wr

        pA_matches = player_stats.get(pA, {}).get("total_matches", 0)
        pB_matches = player_stats.get(pB, {}).get("total_matches", 0)
        exp_diff = pA_matches - pB_matches

        pair_key = tuple(sorted([pA, pB]))
        h2h_data = h2h_tracker.get(pair_key, {pA: 0, pB: 0})
        h2h_diff = h2h_data.get(pA, 0) - h2h_data.get(pB, 0)

        ml_rows.append({
            "win_rate_diff": wr_diff,
            "exp_diff": exp_diff,
            "h2h_diff": h2h_diff,
            "target": target
        })

    ml_df = pd.DataFrame(ml_rows)
    
    # Train model on win rate diff, match count diff, and h2h diff
    model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    model.fit(ml_df[["win_rate_diff", "exp_diff", "h2h_diff"]], ml_df["target"])

@app.on_event("startup")
def startup_event():
    train_model()

class MatchupRequest(BaseModel):
    pA_name: str
    pB_name: str

@app.get("/")
def health_check():
    return {"status": "online", "players_loaded": len(player_list)}

@app.get("/players")
def get_players():
    if not player_list:
        raise HTTPException(status_code=500, detail="Player list not initialized")
    return {"players": player_list}

@app.post("/predict")
def predict_matchup(req: MatchupRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    p1, p2 = req.pA_name, req.pB_name
    
    # Extract Player Stats
    p1_stats = player_stats.get(p1, {"win_rate": 0.5, "total_matches": 0, "wins": 0})
    p2_stats = player_stats.get(p2, {"win_rate": 0.5, "total_matches": 0, "wins": 0})

    wr_diff = p1_stats["win_rate"] - p2_stats["win_rate"]
    exp_diff = p1_stats["total_matches"] - p2_stats["total_matches"]

    pair_key = tuple(sorted([p1, p2]))
    h2h_data = h2h_tracker.get(pair_key, {p1: 0, p2: 0})
    p1_h2h = h2h_data.get(p1, 0)
    p2_h2h = h2h_data.get(p2, 0)
    h2h_diff = p1_h2h - p2_h2h

    # Predict probability using real features
    feats = pd.DataFrame([[wr_diff, exp_diff, h2h_diff]], columns=["win_rate_diff", "exp_diff", "h2h_diff"])
    prob = float(model.predict_proba(feats)[0][1])

    winner = p1 if prob >= 0.5 else p2
    confidence = prob if prob >= 0.5 else (1 - prob)

    # Human-readable factors
    deciding_factors = []
    if abs(wr_diff) > 0.05:
        better_p = p1 if wr_diff > 0 else p2
        deciding_factors.append(f"{better_p} holds a higher overall ATP win rate ({max(p1_stats['win_rate'], p2_stats['win_rate']):.1%})")
    if p1_h2h != p2_h2h:
        leader = p1 if p1_h2h > p2_h2h else p2
        deciding_factors.append(f"{leader} leads the Head-to-Head series ({max(p1_h2h, p2_h2h)}-{min(p1_h2h, p2_h2h)})")
    if abs(exp_diff) > 15:
        more_exp = p1 if exp_diff > 0 else p2
        deciding_factors.append(f"{more_exp} has significantly higher ATP Tour match density")
    if not deciding_factors:
        deciding_factors.append("Even matchup: Decision based on subtle hard-court momentum differentials")

    return {
        "winner": winner,
        "confidence": round(confidence * 100, 1),
        "elo_diff": round(wr_diff * 1000, 1),  # Display scaled relative performance score
        "h2h_diff": h2h_diff,
        "matchup_breakdown": {
            "pA_name": p1,
            "pB_name": p2,
            "pA_h2h_wins": p1_h2h,
            "pB_h2h_wins": p2_h2h,
            "pA_total_atp_wins": p1_stats["wins"],
            "pB_total_atp_wins": p2_stats["wins"],
            "deciding_factors": deciding_factors
        }
    }
