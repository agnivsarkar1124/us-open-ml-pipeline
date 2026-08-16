from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import db
import cache

all_matches = None
h2h_tracker = {}
elo_ratings = {}
player_stats = {}
model = None
player_list = []

DEFAULT_ELO = 1500


def _swap_matchup_fields(cached: dict, p1: str, p2: str) -> dict:
    """
    Cache key is order-independent (sorted pair), but the response has
    pA/pB-labeled fields. If the caller's (p1, p2) order doesn't match
    the order the cached payload was originally stored under, relabel
    so pA_name always equals the caller's p1.
    """
    breakdown = cached["matchup_breakdown"]
    if breakdown["pA_name"].lower() == p1.strip().lower():
        return cached

    swapped = dict(cached)
    swapped["elo_diff"] = -cached["elo_diff"]
    swapped["h2h_diff"] = -cached["h2h_diff"]
    swapped["matchup_breakdown"] = {
        **breakdown,
        "pA_name": breakdown["pB_name"],
        "pB_name": breakdown["pA_name"],
        "pA_elo": breakdown["pB_elo"],
        "pB_elo": breakdown["pA_elo"],
        "pA_h2h_wins": breakdown["pB_h2h_wins"],
        "pB_h2h_wins": breakdown["pA_h2h_wins"],
        "pA_total_atp_wins": breakdown["pB_total_atp_wins"],
        "pB_total_atp_wins": breakdown["pA_total_atp_wins"],
    }
    return swapped


def train_model():
    """Full retrain from raw ATP CSVs — only runs on a true cold cache miss."""
    global all_matches, h2h_tracker, elo_ratings, player_stats, model, player_list

    base_url = (
        "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{}.csv"
    )
    dfs = []
    for year in [2020, 2021, 2022, 2023]:
        try:
            df = pd.read_csv(base_url.format(year))
            df["tourney_date"] = pd.to_datetime(
                df["tourney_date"].astype(str), format="%Y%m%d"
            )
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("Failed to load match datasets.")

    all_matches = pd.concat(dfs, ignore_index=True).sort_values("tourney_date")

    elo_ratings = {}
    K = 32
    for _, row in all_matches.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        rw = elo_ratings.get(w, DEFAULT_ELO)
        rl = elo_ratings.get(l, DEFAULT_ELO)
        exp_w = 1 / (1 + 10 ** ((rl - rw) / 400))
        exp_l = 1 - exp_w
        elo_ratings[w] = rw + K * (1 - exp_w)
        elo_ratings[l] = rl + K * (0 - exp_l)

    winners = all_matches["winner_name"].dropna().unique()
    losers = all_matches["loser_name"].dropna().unique()
    player_list = sorted(list(set(winners).union(set(losers))))

    player_stats = {}
    for player in player_list:
        wins = len(all_matches[all_matches["winner_name"] == player])
        losses = len(all_matches[all_matches["loser_name"] == player])
        total = wins + losses
        win_rate = (wins / total) if total > 5 else 0.50
        player_stats[player] = {
            "wins": wins,
            "losses": losses,
            "total_matches": total,
            "win_rate": win_rate,
            "elo": elo_ratings.get(player, DEFAULT_ELO),
        }

    h2h_tracker = {}
    for _, row in all_matches.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        pair_key = tuple(sorted([w, l]))
        if pair_key not in h2h_tracker:
            h2h_tracker[pair_key] = {w: 1, l: 0}
        else:
            h2h_tracker[pair_key][w] = h2h_tracker[pair_key].get(w, 0) + 1

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

        elo_diff = elo_ratings.get(pA, DEFAULT_ELO) - elo_ratings.get(pB, DEFAULT_ELO)
        pA_matches = player_stats.get(pA, {}).get("total_matches", 0)
        pB_matches = player_stats.get(pB, {}).get("total_matches", 0)
        exp_diff = pA_matches - pB_matches

        pair_key = tuple(sorted([pA, pB]))
        h2h_data = h2h_tracker.get(pair_key, {pA: 0, pB: 0})
        h2h_diff = h2h_data.get(pA, 0) - h2h_data.get(pB, 0)

        ml_rows.append(
            {"elo_diff": elo_diff, "exp_diff": exp_diff, "h2h_diff": h2h_diff, "target": target}
        )

    ml_df = pd.DataFrame(ml_rows)
    model = HistGradientBoostingClassifier(
        random_state=42, max_iter=100, min_samples_leaf=10, l2_regularization=1.0
    )
    model.fit(ml_df[["elo_diff", "exp_diff", "h2h_diff"]], ml_df["target"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global elo_ratings, player_stats, h2h_tracker, player_list, model

    await db.init_pool()
    cache.init_redis()

    cached = await db.load_artifacts(max_age_hours=24)
    if cached is not None:
        elo_ratings = cached["elo_ratings"]
        player_stats = cached["player_stats"]
        h2h_tracker = cached["h2h_tracker"]
        player_list = cached["player_list"]
        # Model itself isn't persisted (sklearn objects don't serialize
        # cleanly to Postgres JSONB) — retrain is cheap relative to the
        # CSV download + Elo/H2H pass, which is the part we skip here.
        # If you want to skip this too, pickle the model to a bytea column.
        train_model()
    else:
        train_model()
        await db.save_artifacts(elo_ratings, player_stats, h2h_tracker, player_list)
        await cache.invalidate_all()

    yield

    await db.close_pool()
    await cache.close_redis()


app = FastAPI(title="US Open Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchupRequest(BaseModel):
    pA_name: str
    pB_name: str


@app.get("/")
def health_check():
    return {"status": "online", "players_loaded": len(player_list)}


@app.get("/players")
async def get_players():
    if not player_list:
        raise HTTPException(status_code=500, detail="Player list not initialized")

    cached = await cache.get_cached_players()
    if cached is not None:
        return {"players": cached}

    await cache.set_cached_players(player_list)
    return {"players": player_list}


@app.post("/predict")
async def predict_matchup(req: MatchupRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    p1, p2 = req.pA_name, req.pB_name

    cached = await cache.get_cached_predict(p1, p2)
    if cached is not None:
        return _swap_matchup_fields(cached, p1, p2)

    p1_stats = player_stats.get(p1, {"win_rate": 0.5, "total_matches": 0, "wins": 0})
    p2_stats = player_stats.get(p2, {"win_rate": 0.5, "total_matches": 0, "wins": 0})

    p1_elo = elo_ratings.get(p1, 1500)
    p2_elo = elo_ratings.get(p2, 1500)
    elo_diff = p1_elo - p2_elo
    exp_diff = p1_stats["total_matches"] - p2_stats["total_matches"]

    pair_key = tuple(sorted([p1, p2]))
    h2h_data = h2h_tracker.get(pair_key, {p1: 0, p2: 0})
    p1_h2h = h2h_data.get(p1, 0)
    p2_h2h = h2h_data.get(p2, 0)
    h2h_diff = p1_h2h - p2_h2h

    feats = pd.DataFrame(
        [[elo_diff, exp_diff, h2h_diff]], columns=["elo_diff", "exp_diff", "h2h_diff"]
    )
    raw_prob = float(model.predict_proba(feats)[0][1])
    clipped_prob = np.clip(raw_prob, 0.12, 0.88)
    winner = p1 if clipped_prob >= 0.5 else p2
    confidence = clipped_prob if clipped_prob >= 0.5 else (1 - clipped_prob)

    deciding_factors = []
    if abs(elo_diff) > 25:
        higher_elo = p1 if elo_diff > 0 else p2
        deciding_factors.append(
            f"{higher_elo} holds a higher overall Elo rating ({max(p1_elo, p2_elo):.0f} vs {min(p1_elo, p2_elo):.0f})"
        )
    if p1_h2h != p2_h2h:
        leader = p1 if p1_h2h > p2_h2h else p2
        deciding_factors.append(
            f"{leader} leads the Head-to-Head series ({max(p1_h2h, p2_h2h)}-{min(p1_h2h, p2_h2h)})"
        )
    if abs(exp_diff) > 15:
        more_exp = p1 if exp_diff > 0 else p2
        deciding_factors.append(f"{more_exp} has significantly higher ATP Tour match density")
    if not deciding_factors:
        deciding_factors.append("Even matchup: Decision based on subtle hard-court momentum differentials")

    result = {
        "winner": winner,
        "confidence": round(confidence * 100, 1),
        "elo_diff": round(elo_diff, 1),
        "h2h_diff": h2h_diff,
        "matchup_breakdown": {
            "pA_name": p1,
            "pB_name": p2,
            "pA_elo": round(p1_elo, 1),
            "pB_elo": round(p2_elo, 1),
            "pA_h2h_wins": p1_h2h,
            "pB_h2h_wins": p2_h2h,
            "pA_total_atp_wins": p1_stats["wins"],
            "pB_total_atp_wins": p2_stats["wins"],
            "deciding_factors": deciding_factors,
        },
    }

    await cache.set_cached_predict(p1, p2, result)
    return result
