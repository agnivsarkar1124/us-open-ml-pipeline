"""
Redis layer: connection pooling + response caching for the hot endpoints.

/predict is a pure function of (pA_name, pB_name) given the current trained
state, and /players barely changes — both are ideal cache candidates.
"""

import json
import os

import redis.asyncio as redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# TTLs: predictions can live longer since the model only retrains once a day
# (see db.load_artifacts max_age_hours); players list is smaller and cheaper
# to recompute so a shorter TTL is fine too, mostly just to smooth spikes.
PREDICT_TTL_SECONDS = 60 * 60 * 6   # 6 hours
PLAYERS_TTL_SECONDS = 60 * 60       # 1 hour

_redis_client: redis.Redis | None = None

# In-process counters for /system-status. These reset on every restart —
# that's fine, the point is to show cache behavior since the last deploy,
# not a permanent metric store.
stats = {"predict_hits": 0, "predict_misses": 0, "players_hits": 0, "players_misses": 0}


def init_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,
            decode_responses=True,
        )
        _redis_client = redis.Redis(connection_pool=pool)
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def get_redis() -> redis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized — call init_redis() on startup")
    return _redis_client


def predict_cache_key(pA_name: str, pB_name: str) -> str:
    # Sort so pA/pB order doesn't create duplicate cache entries for the
    # same matchup — but keep the ORIGINAL request order in the response
    # you return (the cached payload already carries correct pA/pB labels
    # since the caller in main.py stores it under the original order too).
    a, b = sorted([pA_name.strip().lower(), pB_name.strip().lower()])
    return f"predict:{a}::{b}"


async def get_cached_predict(pA_name: str, pB_name: str) -> dict | None:
    client = get_redis()
    raw = await client.get(predict_cache_key(pA_name, pB_name))
    if raw:
        stats["predict_hits"] += 1
        return json.loads(raw)
    stats["predict_misses"] += 1
    return None


async def set_cached_predict(pA_name: str, pB_name: str, payload: dict) -> None:
    client = get_redis()
    await client.set(
        predict_cache_key(pA_name, pB_name),
        json.dumps(payload),
        ex=PREDICT_TTL_SECONDS,
    )


async def get_cached_players() -> list | None:
    client = get_redis()
    raw = await client.get("players:list")
    if raw:
        stats["players_hits"] += 1
        return json.loads(raw)
    stats["players_misses"] += 1
    return None


async def set_cached_players(players: list) -> None:
    client = get_redis()
    await client.set("players:list", json.dumps(players), ex=PLAYERS_TTL_SECONDS)


async def invalidate_all() -> None:
    """Call after a fresh training run so stale predictions don't linger."""
    client = get_redis()
    async for key in client.scan_iter(match="predict:*"):
        await client.delete(key)
    await client.delete("players:list")
