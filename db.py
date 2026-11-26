"""
PostgreSQL layer: connection pooling + persistence of the trained model.

After a full training run, the fitted ensemble, the feature builder's player
state and the roster are pickled into a single-row snapshot table. On the next
start, a fresh snapshot (default: under 24 h old) is loaded instead of
re-downloading 16 seasons of ATP data and retraining.

Uses psycopg (v3) with psycopg_pool: prebuilt wheels via the
`psycopg[binary]` extra and an async pooled API.

Persistence is optional: if DATABASE_URL is unset or Postgres can't be
reached, the app logs a warning and trains on every cold start instead.
A dead database must never stop the API from booting.

Note: snapshots are unpickled, so only point DATABASE_URL at a database
this app alone writes to.
"""

import logging
import os
import pickle

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

log = logging.getLogger("uvicorn.error")

DATABASE_URL = os.environ.get("DATABASE_URL")  # postgresql://user:pass@host:port/db
CONNECT_TIMEOUT_SECONDS = 10
SNAPSHOT_VERSION = 2  # bump when the snapshot's structure changes

DB_ERRORS = (psycopg.Error, PoolTimeout, OSError)

_pool: AsyncConnectionPool | None = None


async def init_pool() -> bool:
    """Open the pool and ensure the schema. Returns True if Postgres is usable."""
    global _pool
    if _pool is not None:
        return True

    if not DATABASE_URL:
        log.warning("DATABASE_URL not set — running without Postgres persistence")
        return False

    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        timeout=CONNECT_TIMEOUT_SECONDS,
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)
        await _ensure_schema(pool)
    except DB_ERRORS as exc:
        log.warning("Postgres unavailable (%s) — running without persistence", exc)
        await pool.close()
        return False

    _pool = pool
    log.info("Postgres pool connected")
    return True


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def is_enabled() -> bool:
    return _pool is not None


async def _ensure_schema(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_snapshot (
                id SMALLINT PRIMARY KEY DEFAULT 1,
                version SMALLINT NOT NULL,
                payload BYTEA NOT NULL,
                trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT single_row CHECK (id = 1)
            );
            """
        )


async def save_snapshot(state: dict) -> None:
    """Upsert the trained state as a single-row snapshot (no-op if disabled)."""
    if _pool is None:
        return
    try:
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        async with _pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO model_snapshot (id, version, payload, trained_at)
                VALUES (1, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    version = EXCLUDED.version,
                    payload = EXCLUDED.payload,
                    trained_at = EXCLUDED.trained_at;
                """,
                (SNAPSHOT_VERSION, payload),
            )
        log.info("Saved model snapshot (%.1f MB)", len(payload) / 1e6)
    except DB_ERRORS as exc:
        log.warning("Saving model snapshot failed (%s)", exc)


async def load_snapshot(max_age_hours: int = 24) -> dict | None:
    """Return the stored state if present, current-version and fresh enough, else None."""
    if _pool is None:
        return None
    try:
        async with _pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT payload FROM model_snapshot
                    WHERE id = 1 AND version = %s
                      AND trained_at > now() - (%s || ' hours')::interval;
                    """,
                    (SNAPSHOT_VERSION, str(max_age_hours)),
                )
                row = await cur.fetchone()
    except DB_ERRORS as exc:
        log.warning("Loading model snapshot failed (%s)", exc)
        return None
    if row is None:
        return None
    try:
        state = pickle.loads(row["payload"])
    except Exception as exc:  # e.g. snapshot from an incompatible library version
        log.warning("Model snapshot could not be unpickled (%s) — retraining", exc)
        return None
    log.info("Loaded model snapshot — skipping training")
    return state


async def get_trained_at() -> str | None:
    """Lightweight check for /system-status — just the timestamp, no payload."""
    if _pool is None:
        return None
    try:
        async with _pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT trained_at FROM model_snapshot WHERE id = 1;")
                row = await cur.fetchone()
    except DB_ERRORS as exc:
        log.warning("Postgres status check failed (%s)", exc)
        return None
    return row["trained_at"].isoformat() if row else None
