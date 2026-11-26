"""
Leakage-free feature engineering for ATP match prediction.

`FeatureBuilder` walks matches in chronological order. For each match it
first snapshots both players' state (ratings, form, serve/return quality,
workload), and only THEN updates that state with the result. Every feature
is therefore something you could have known before the first ball.

The same builder is used by evaluate.py (backtesting) and main.py (serving),
so train-time and serve-time features can't drift apart.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{}.csv"
DEFAULT_ELO = 1500.0
ROUND_ORDER = {r: i for i, r in enumerate(
    ["Q1", "Q2", "Q3", "RR", "R128", "R64", "R32", "R16", "QF", "SF", "BR", "F"]
)}
MAIN_TOUR_LEVELS = {"G", "M", "A", "F"}  # Slams, Masters, ATP 250/500, Finals

# Prior means for serve / return points won (tour averages) and EWMA weight.
SPW_PRIOR, RPW_PRIOR, EW_ALPHA = 0.63, 0.37, 0.06


def k_factor(n_matches: int) -> float:
    """Decaying K: new players move fast, veterans settle (FiveThirtyEight-style)."""
    return 250.0 / (n_matches + 5) ** 0.4


def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


@dataclass
class PlayerState:
    elo_fixed: float = DEFAULT_ELO       # original K=32 Elo (old baseline)
    elo: float = DEFAULT_ELO             # dynamic-K, all surfaces
    elo_hard: float = DEFAULT_ELO        # dynamic-K, hard courts only
    n: int = 0
    n_hard: int = 0
    spw: float = SPW_PRIOR               # EWMA serve points won
    rpw: float = RPW_PRIOR               # EWMA return points won
    n_stats: int = 0
    last_date: pd.Timestamp | None = None
    results: deque = field(default_factory=lambda: deque(maxlen=10))
    elo_hist: deque = field(default_factory=lambda: deque(maxlen=80))  # (date, elo)
    tourney_id: str | None = None
    tourney_minutes: float = 0.0


def _safe_log(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return math.log(x) if x > 0 else math.nan


class FeatureBuilder:
    # A-minus-B differential features fed to the model
    FEATURES = [
        "elo_diff", "elo_hard_diff", "form_diff", "win10_diff",
        "log_rank_diff", "log_pts_diff", "spw_diff", "rpw_diff",
        "log_exp_diff", "h2h_diff", "age_diff", "rest_diff", "fatigue_diff",
    ]
    # +1: larger A-minus-B value should only ever help A; -1: only hurt; 0: free
    MONOTONIC = {
        "elo_diff": 1, "elo_hard_diff": 1, "form_diff": 0, "win10_diff": 1,
        "log_rank_diff": -1, "log_pts_diff": 1, "spw_diff": 1, "rpw_diff": 1,
        "log_exp_diff": 0, "h2h_diff": 1, "age_diff": 0, "rest_diff": 0, "fatigue_diff": 0,
    }

    def __init__(self):
        self.players: dict[str, PlayerState] = {}
        self.h2h: dict[tuple[str, str], dict[str, int]] = {}
        self.last_date: pd.Timestamp | None = None

    def _p(self, name: str) -> PlayerState:
        if name not in self.players:
            self.players[name] = PlayerState()
        return self.players[name]

    # ------------------------------------------------------------------ snapshot
    def _player_snapshot(self, name: str, date, tourney_id, rank, pts, age) -> dict:
        s = self._p(name)
        # Form: Elo change versus ~90 days ago
        past = [e for d, e in s.elo_hist if (date - d).days >= 90]
        form = s.elo - past[-1] if past else 0.0
        rest = (date - s.last_date).days if s.last_date is not None else 60
        return {
            "elo_fixed": s.elo_fixed,
            "elo": s.elo,
            "elo_hard": s.elo_hard,
            "form": form,
            "win10": (sum(s.results) / len(s.results)) if s.results else 0.5,
            "log_rank": _safe_log(rank),
            "log_pts": _safe_log(pts),
            "spw": s.spw,
            "rpw": s.rpw,
            "log_exp": math.log1p(s.n),
            "n": s.n,
            "age": float(age) if pd.notna(age) else math.nan,
            "rest": min(rest, 60),
            "fatigue": s.tourney_minutes if s.tourney_id == tourney_id else 0.0,
        }

    def matchup_features(self, a: str, b: str, date=None, tourney_id=None,
                         a_rank=None, b_rank=None, a_pts=None, b_pts=None,
                         a_age=None, b_age=None) -> dict:
        """A-minus-B feature row for a (possibly hypothetical) match between a and b."""
        date = pd.Timestamp(date) if date is not None else (self.last_date or pd.Timestamp.today())
        sa = self._player_snapshot(a, date, tourney_id, a_rank, a_pts, a_age)
        sb = self._player_snapshot(b, date, tourney_id, b_rank, b_pts, b_age)
        pair = tuple(sorted([a, b]))
        rec = self.h2h.get(pair, {})
        return {
            "elo_fixed_diff": sa["elo_fixed"] - sb["elo_fixed"],
            "elo_diff": sa["elo"] - sb["elo"],
            "elo_hard_diff": sa["elo_hard"] - sb["elo_hard"],
            "form_diff": sa["form"] - sb["form"],
            "win10_diff": sa["win10"] - sb["win10"],
            "log_rank_diff": sa["log_rank"] - sb["log_rank"],
            "log_pts_diff": sa["log_pts"] - sb["log_pts"],
            "spw_diff": sa["spw"] - sb["spw"],
            "rpw_diff": sa["rpw"] - sb["rpw"],
            "log_exp_diff": sa["log_exp"] - sb["log_exp"],
            "h2h_diff": rec.get(a, 0) - rec.get(b, 0),
            "age_diff": sa["age"] - sb["age"],
            "rest_diff": sa["rest"] - sb["rest"],
            "fatigue_diff": sa["fatigue"] - sb["fatigue"],
            # raw values, for explanations and the v1 baseline
            "exp_diff": sa["n"] - sb["n"],
            "pA_matches": sa["n"], "pB_matches": sb["n"],
            "pA_spw": sa["spw"], "pB_spw": sb["spw"],
            "pA_rpw": sa["rpw"], "pB_rpw": sb["rpw"],
            "pA_elo": sa["elo"], "pB_elo": sb["elo"],
            "pA_elo_hard": sa["elo_hard"], "pB_elo_hard": sb["elo_hard"],
            "pA_h2h": rec.get(a, 0), "pB_h2h": rec.get(b, 0),
        }

    # ------------------------------------------------------------------ update
    def _update(self, r) -> None:
        w, l = r.winner_name, r.loser_name
        sw, sl = self._p(w), self._p(l)
        date = r.tourney_date

        # fixed-K Elo
        e = expected(sw.elo_fixed, sl.elo_fixed)
        sw.elo_fixed += 32 * (1 - e)
        sl.elo_fixed -= 32 * (1 - e)
        # dynamic-K Elo (each player's own K)
        e = expected(sw.elo, sl.elo)
        sw.elo += k_factor(sw.n) * (1 - e)
        sl.elo -= k_factor(sl.n) * (1 - e)
        if r.surface == "Hard":
            e = expected(sw.elo_hard, sl.elo_hard)
            sw.elo_hard += k_factor(sw.n_hard) * (1 - e)
            sl.elo_hard -= k_factor(sl.n_hard) * (1 - e)
            sw.n_hard += 1
            sl.n_hard += 1

        # serve / return quality from this match's point stats, if recorded
        try:
            w_sv, l_sv = float(r.w_svpt), float(r.l_svpt)
            w_won = float(r.w_1stWon) + float(r.w_2ndWon)
            l_won = float(r.l_1stWon) + float(r.l_2ndWon)
            if w_sv > 0 and l_sv > 0:
                for s, sv_pct, rt_pct in ((sw, w_won / w_sv, 1 - l_won / l_sv),
                                          (sl, l_won / l_sv, 1 - w_won / w_sv)):
                    s.spw += EW_ALPHA * (sv_pct - s.spw)
                    s.rpw += EW_ALPHA * (rt_pct - s.rpw)
                    s.n_stats += 1
        except (TypeError, ValueError):
            pass

        minutes = float(r.minutes) if pd.notna(r.minutes) else 0.0
        for s, won in ((sw, 1), (sl, 0)):
            s.n += 1
            s.results.append(won)
            s.elo_hist.append((date, s.elo))
            s.last_date = date
            if s.tourney_id != r.tourney_id:
                s.tourney_id, s.tourney_minutes = r.tourney_id, 0.0
            s.tourney_minutes += minutes

        pair = tuple(sorted([w, l]))
        self.h2h.setdefault(pair, {})
        self.h2h[pair][w] = self.h2h[pair].get(w, 0) + 1
        self.last_date = date

    # ------------------------------------------------------------------ driver
    def process(self, df: pd.DataFrame, record_from_year: int, seed: int = 42) -> pd.DataFrame:
        """Return one pre-match feature row per match from `record_from_year` on."""
        self.last_date = None
        rng = np.random.default_rng(seed)
        rows = []
        for r in df.itertuples(index=False):
            w, l = r.winner_name, r.loser_name
            if not isinstance(w, str) or not isinstance(l, str):
                continue
            if r.tourney_date.year >= record_from_year:
                swap = rng.random() > 0.5
                a, b = (l, w) if swap else (w, l)
                ra, rb = (r.loser_rank, r.winner_rank) if swap else (r.winner_rank, r.loser_rank)
                pa, pb = ((r.loser_rank_points, r.winner_rank_points) if swap
                          else (r.winner_rank_points, r.loser_rank_points))
                aa, ab = (r.loser_age, r.winner_age) if swap else (r.winner_age, r.loser_age)
                feats = self.matchup_features(a, b, r.tourney_date, r.tourney_id, ra, rb, pa, pb, aa, ab)
                rows.append({
                    "date": r.tourney_date,
                    "year": r.tourney_date.year,
                    "tourney_name": r.tourney_name,
                    "tourney_level": r.tourney_level,
                    "surface": r.surface,
                    "best_of": r.best_of,
                    "round": r.round,
                    "pA": a,
                    "pB": b,
                    "target": 0 if swap else 1,
                    **feats,
                })
            self._update(r)
        return pd.DataFrame(rows)


USED_COLUMNS = {
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", "match_num",
    "winner_name", "loser_name", "winner_rank", "loser_rank", "winner_rank_points",
    "loser_rank_points", "winner_age", "loser_age", "best_of", "round", "minutes",
    "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon",
}


def load_matches(years, data_dir: str | None = None) -> pd.DataFrame:
    from pathlib import Path

    dfs = []
    for y in years:
        src = Path(data_dir) / f"atp_matches_{y}.csv" if data_dir else BASE_URL.format(y)
        try:
            dfs.append(pd.read_csv(src, usecols=lambda c: c in USED_COLUMNS, low_memory=False))
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No match data could be loaded")
    df = pd.concat(dfs, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df["round_idx"] = df["round"].map(ROUND_ORDER).fillna(5)
    return df.sort_values(["tourney_date", "tourney_id", "round_idx", "match_num"]).reset_index(drop=True)


def is_us_open(name) -> bool:
    return "us open" in str(name).lower()
