"""
LangChain tools that wrap the US Open prediction API.

The agent never touches the model directly: every number it can use comes
from one of these HTTP calls, which is what makes grounding checkable.
"""

from __future__ import annotations

import difflib
import os

import httpx
from langchain_core.tools import tool

DEFAULT_API_URL = "https://two026-us-open-ml-predictor.onrender.com"
_client: httpx.Client | None = None


def set_client(client: httpx.Client) -> None:
    """Point the tools at a specific API client (e.g. FastAPI's TestClient in tests)."""
    global _client
    _client = client


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=os.environ.get("US_OPEN_API_URL", DEFAULT_API_URL),
            timeout=httpx.Timeout(60.0, connect=10.0),  # Render free tier cold starts are slow
        )
    return _client


def _get(path: str, **params) -> dict:
    r = get_client().get(path, params=params or None)
    if r.status_code == 404:
        return {"error": r.json().get("detail", "not found"),
                "hint": "Use search_players to find the exact name."}
    r.raise_for_status()
    return r.json()


@tool
def search_players(query: str) -> dict:
    """Find exact player names in the dataset. Use this when a name is partial,
    misspelled or lowercase (e.g. 'sinner', 'de minaur') before calling other tools."""
    players = _get("/players")["players"]
    tokens = query.strip().lower().split()
    hits = [p for p in players if all(t in p.lower() for t in tokens)]  # any word order
    if not hits:
        hits = difflib.get_close_matches(query, players, n=5, cutoff=0.6)
    return {"query": query, "matches": hits[:10]}


@tool
def predict_match(player_a: str, player_b: str) -> dict:
    """Model prediction for a best-of-five hard-court (US Open) match: predicted
    winner, win probability, Elo ratings, serve/return stats and deciding factors.
    Always call this for any matchup question."""
    r = get_client().post("/predict", json={"pA_name": player_a, "pB_name": player_b})
    if r.status_code == 422:
        return {"error": r.json().get("detail"), "hint": "Use search_players to find exact names."}
    r.raise_for_status()
    out = r.json()
    out.pop("response_time_ms", None)  # transport detail, not a tennis fact
    out.pop("cached", None)
    return out


@tool
def player_profile(name: str) -> dict:
    """One player's current snapshot: overall and hard-court Elo, ATP rank and points,
    age, serve/return points won %, recent form (Elo change over 90 days, wins in last 10)."""
    return _get(f"/players/{name}/profile")


@tool
def head_to_head(player_a: str, player_b: str) -> dict:
    """Career head-to-head record between two players in the dataset (2010 onward)."""
    return _get("/h2h", p1=player_a, p2=player_b)


TOOLS = [search_players, predict_match, player_profile, head_to_head]
TOOL_NAMES = {t.name for t in TOOLS}
