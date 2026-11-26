"""
Redis layer: connection pooling + response caching for the hot endpoints.

/predict is a pure function of (pA_name, pB_name) given the current trained
state, and /players barely changes — both are ideal cache candidates.

The cache is strictly best-effort: if REDIS_URL is unset, Redis is
unreachable at startup, or a call fails mid-flight, every helper here
degrades to a cache miss / no-op instead of raising. The API must never
go down because the cache did.
"""

import json
import logging
import os

import redis.asyncio as redis
from redis.exceptions import RedisError

log = logging.getLogger("uvicorn.error")

REDIS_URL = os.environ.get("REDIS_URL")

# TTLs: predictions can live longer since the model only retrains once a day
# (see db.load_artifacts max_age_hours); players list is smaller and cheaper
# to recompute so a shorter TTL is fine too, mostly just to smooth spikes.
PREDICT_TTL_SECONDS = 60 * 60 * 6   # 6 hours
PLAYERS_TTL_SECONDS = 60 * 60       # 1 hour

# Bump when the model or response schema changes so old payloads are never served.
MODEL_VERSION = "v2"

# Short timeouts so a dead/slow Redis costs milliseconds, not a hung request.
CONNECT_TIMEOUT_SECONDS = 2
SOCKET_TIMEOUT_SECONDS = 2

_redis_client: redis.Redis | None = None
enabled = False  # True only after a successful startup PING

# In-process counters for /system-status. These reset on every restart —
# that's fine, the point is to show cache behavior since the last deploy,
# not a permanent metric store.
stats = {
    "predict_hits": 0,
    "predict_misses": 0,
    "players_hits": 0,
    "players_misses": 0,
    "errors": 0,
}


async def init_redis() -> bool:
    """Create the pooled client and PING it. Returns True if the cache is usable."""
    global _redis_client, enabled

    if not REDIS_URL:
        log.warning("REDIS_URL not set — running without Redis cache")
        enabled = False
        return False

    try:
        pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,
            decode_responses=True,
            socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        _redis_client = redis.Redis(connection_pool=pool)
        await _redis_client.ping()
        enabled = True
        log.info("Redis cache connected")
    except (RedisError, OSError, ValueError) as exc:
        log.warning("Redis unavailable (%s) — running without cache", exc)
        await close_redis()
        enabled = False

    return enabled


async def close_redis() -> None:
    global _redis_client, enabled
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except (RedisError, OSError):
            pass
        _redis_client = None
    enabled = False


def _client() -> redis.Redis | None:
    return _redis_client if enabled else None


def _record_error(op: str, exc: Exception) -> None:
    stats["errors"] += 1
    log.warning("Redis %s failed (%s) — treating as cache miss", op, exc)


def predict_cache_key(pA_name: str, pB_name: str) -> str:
    # Sort so pA/pB order doesn't create duplicate cache entries for the
    # same matchup. main._swap_matchup_fields relabels a cached payload
    # back into the caller's original pA/pB order.
    a, b = sorted([pA_name.strip().lower(), pB_name.strip().lower()])
    return f"predict:{MODEL_VERSION}:{a}::{b}"


async def get_cached_predict(pA_name: str, pB_name: str) -> dict | None:
    client = _client()
    if client is None:
        return None
    try:
        raw = await client.get(predict_cache_key(pA_name, pB_name))
    except (RedisError, OSError) as exc:
        _record_error("GET predict", exc)
        return None
    if raw:
        stats["predict_hits"] += 1
        return json.loads(raw)
    stats["predict_misses"] += 1
    return None


async def set_cached_predict(pA_name: str, pB_name: str, payload: dict) -> None:
    client = _client()
    if client is None:
        return
    try:
        await client.set(
            predict_cache_key(pA_name, pB_name),
            json.dumps(payload),
            ex=PREDICT_TTL_SECONDS,
        )
    except (RedisError, OSError) as exc:
        _record_error("SET predict", exc)


async def get_cached_players() -> list | None:
    client = _client()
    if client is None:
        return None
    try:
        raw = await client.get(f"players:{MODEL_VERSION}")
    except (RedisError, OSError) as exc:
        _record_error("GET players", exc)
        return None
    if raw:
        stats["players_hits"] += 1
        return json.loads(raw)
    stats["players_misses"] += 1
    return None


async def set_cached_players(players: list) -> None:
    client = _client()
    if client is None:
        return
    try:
        await client.set(f"players:{MODEL_VERSION}", json.dumps(players), ex=PLAYERS_TTL_SECONDS)
    except (RedisError, OSError) as exc:
        _record_error("SET players", exc)


async def invalidate_all() -> None:
    """Call after a fresh training run so stale predictions don't linger."""
    client = _client()
    if client is None:
        return
    try:
        async for key in client.scan_iter(match="predict:*", count=500):
            await client.delete(key)
        await client.delete(f"players:{MODEL_VERSION}")
    except (RedisError, OSError) as exc:
        _record_error("invalidate", exc)
