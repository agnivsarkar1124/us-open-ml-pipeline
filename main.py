import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cache
import db
from features import FeatureBuilder, load_matches
from model import MatchModel, add_model_columns, training_rows

DATA_YEARS = range(2010, 2026)
ROSTER_MIN_MATCHES = 10   # players shown in the picker need some history...
ROSTER_RECENT_YEARS = 2   # ...and a match in the last two seasons of data

# Everything the API needs to serve predictions; persisted to Postgres as one snapshot.
state: dict = {}


def build_state() -> dict:
    """Download match data, build leakage-free features, and fit the ensemble."""
    matches = load_matches(DATA_YEARS)
    builder = FeatureBuilder()
    rows = add_model_columns(builder.process(matches, record_from_year=2014))
    model = MatchModel().fit(training_rows(rows))

    # Latest known rank / points / age per player, for hypothetical matchups.
    info = {}
    for side in ("winner", "loser"):
        cols = matches[["tourney_date", f"{side}_name", f"{side}_rank",
                        f"{side}_rank_points", f"{side}_age"]].copy()
        cols.columns = ["date", "name", "rank", "pts", "age"]
        for r in cols.itertuples(index=False):
            if isinstance(r.name, str) and (r.name not in info or r.date >= info[r.name]["date"]):
                info[r.name] = {"date": r.date, "rank": r.rank, "pts": r.pts, "age": r.age}

    as_of = builder.last_date + pd.Timedelta(days=1)
    recent_cutoff = as_of - pd.DateOffset(years=ROSTER_RECENT_YEARS)
    roster = sorted(
        name for name, s in builder.players.items()
        if s.n >= ROSTER_MIN_MATCHES and s.last_date is not None and s.last_date >= recent_cutoff
    )
    return {
        "model": model,
        "builder": builder,
        "info": info,
        "player_list": roster,
        "roster_set": set(roster),
        "as_of": as_of,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state

    # Both are optional: each logs a warning and returns False if unavailable,
    # so a dead database or cache can't crash startup.
    await db.init_pool()
    await cache.init_redis()

    snapshot = await db.load_snapshot(max_age_hours=24)
    if snapshot is not None:
        state = snapshot          # warm start: skip download + training entirely
    else:
        state = await asyncio.to_thread(build_state)
        await db.save_snapshot(state)
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


def _swap_matchup_fields(cached: dict, p1: str) -> dict:
    """
    Cache key is order-independent (sorted pair), but the response has
    pA/pB-labeled fields. If the caller's order differs from the order the
    payload was stored under, relabel so pA_name always equals the caller's p1.
    """
    breakdown = cached["matchup_breakdown"]
    if breakdown["pA_name"].lower() == p1.strip().lower():
        return {**cached, "cached": True}

    swapped = {**cached, "cached": True,
               "elo_diff": -cached["elo_diff"], "h2h_diff": -cached["h2h_diff"]}
    new_breakdown = dict(breakdown)
    for field in ("name", "elo", "hard_elo", "h2h_wins", "serve_pts_won", "return_pts_won"):
        new_breakdown[f"pA_{field}"] = breakdown[f"pB_{field}"]
        new_breakdown[f"pB_{field}"] = breakdown[f"pA_{field}"]
    swapped["matchup_breakdown"] = new_breakdown
    return swapped


def _deciding_factors(f: dict, a: str, b: str) -> list[str]:
    out = []

    def lead(diff):
        return (a, b) if diff > 0 else (b, a)

    if abs(f["elo_diff"]) > 25:
        hi, _ = lead(f["elo_diff"])
        out.append(f"{hi} holds a higher overall Elo rating "
                   f"({max(f['pA_elo'], f['pB_elo']):.0f} vs {min(f['pA_elo'], f['pB_elo']):.0f})")
    if abs(f["elo_hard_diff"]) > 25:
        hi, _ = lead(f["elo_hard_diff"])
        out.append(f"{hi} is rated higher on hard courts "
                   f"({max(f['pA_elo_hard'], f['pB_elo_hard']):.0f} vs {min(f['pA_elo_hard'], f['pB_elo_hard']):.0f})")
    if abs(f["spw_diff"]) > 0.01:
        hi, _ = lead(f["spw_diff"])
        out.append(f"{hi} wins more service points recently "
                   f"({max(f['pA_spw'], f['pB_spw']):.1%} vs {min(f['pA_spw'], f['pB_spw']):.1%})")
    if abs(f["rpw_diff"]) > 0.01:
        hi, _ = lead(f["rpw_diff"])
        out.append(f"{hi} wins more return points recently "
                   f"({max(f['pA_rpw'], f['pB_rpw']):.1%} vs {min(f['pA_rpw'], f['pB_rpw']):.1%})")
    if abs(f["form_diff"]) > 40:
        hi, _ = lead(f["form_diff"])
        out.append(f"{hi} is in better form (larger Elo gain over the last 3 months)")
    if f["pA_h2h"] != f["pB_h2h"]:
        hi, _ = lead(f["pA_h2h"] - f["pB_h2h"])
        out.append(f"{hi} leads the head-to-head ({max(f['pA_h2h'], f['pB_h2h'])}-{min(f['pA_h2h'], f['pB_h2h'])})")
    return out[:4] or ["Evenly matched: ratings and recent serve/return numbers are close"]


@app.get("/")
def health_check():
    return {"status": "online", "players_loaded": len(state.get("player_list", []))}


@app.get("/system-status")
async def system_status():
    """Visible proof the caching/persistence layer is active."""
    return {
        "postgres_connected": db.is_enabled(),
        "postgres_last_trained_at": await db.get_trained_at(),
        "model_trained_at": state.get("trained_at"),
        "data_through": state["as_of"].date().isoformat() if state else None,
        "redis_connected": cache.enabled,
        "redis_stats": cache.stats,
    }


@app.get("/players")
async def get_players():
    if not state.get("player_list"):
        raise HTTPException(status_code=500, detail="Player list not initialized")

    cached = await cache.get_cached_players()
    if cached is not None:
        return {"players": cached}

    await cache.set_cached_players(state["player_list"])
    return {"players": state["player_list"]}


@app.post("/predict")
async def predict_matchup(req: MatchupRequest):
    if not state.get("model"):
        raise HTTPException(status_code=500, detail="Model not trained")

    start = time.perf_counter()
    p1, p2 = req.pA_name.strip(), req.pB_name.strip()
    for p in (p1, p2):
        if p not in state["roster_set"]:
            raise HTTPException(status_code=422, detail=f"Unknown player: {p}")
    if p1 == p2:
        raise HTTPException(status_code=422, detail="Pick two different players")

    cached = await cache.get_cached_predict(p1, p2)
    if cached is not None:
        result = _swap_matchup_fields(cached, p1)
        result["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    builder, info, as_of = state["builder"], state["info"], state["as_of"]

    def years_since(d):
        return (as_of - d).days / 365.25

    ia, ib = info.get(p1, {}), info.get(p2, {})
    f = builder.matchup_features(
        p1, p2, date=as_of, tourney_id=None,
        a_rank=ia.get("rank"), b_rank=ib.get("rank"),
        a_pts=ia.get("pts"), b_pts=ib.get("pts"),
        a_age=(ia["age"] + years_since(ia["date"])) if ia else None,
        b_age=(ib["age"] + years_since(ib["date"])) if ib else None,
    )
    row = add_model_columns(pd.DataFrame([{**f, "best_of": 5}]))  # US Open men's = best of 5
    p_a = float(state["model"].predict_proba(row)[0])
    winner = p1 if p_a >= 0.5 else p2

    result = {
        "winner": winner,
        "confidence": round(max(p_a, 1 - p_a) * 100, 1),
        "elo_diff": round(f["elo_diff"], 1),
        "h2h_diff": int(f["h2h_diff"]),
        "cached": False,
        "matchup_breakdown": {
            "pA_name": p1,
            "pB_name": p2,
            "pA_elo": round(f["pA_elo"], 1),
            "pB_elo": round(f["pB_elo"], 1),
            "pA_hard_elo": round(f["pA_elo_hard"], 1),
            "pB_hard_elo": round(f["pB_elo_hard"], 1),
            "pA_h2h_wins": int(f["pA_h2h"]),
            "pB_h2h_wins": int(f["pB_h2h"]),
            "pA_serve_pts_won": round(f["pA_spw"], 3),
            "pB_serve_pts_won": round(f["pB_spw"], 3),
            "pA_return_pts_won": round(f["pA_rpw"], 3),
            "pB_return_pts_won": round(f["pB_rpw"], 3),
            "deciding_factors": _deciding_factors(f, p1, p2),
        },
    }

    await cache.set_cached_predict(p1, p2, result)
    result["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result
