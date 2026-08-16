"""
PostgreSQL layer: connection pooling + persistence for trained artifacts
(elo_ratings, player_stats, h2h_tracker) so a cold start can skip
re-downloading and re-training against 4 years of ATP CSVs.

Uses asyncpg directly (no ORM) since the access pattern here is a handful
of bulk upserts on startup + bulk reads on cold start — an ORM would just
add overhead for no benefit.
"""

import json
import os

import asyncpg

DATABASE_URL = os.environ["DATABASE_URL"]  # postgres://user:pass@host:port/db

# Tune pool size to your Render/Postgres plan. min_size keeps warm connections
# ready so the first request after idle doesn't pay a connection-setup cost.
_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=2,
            max_size=10,
            max_inactive_connection_lifetime=300,
            command_timeout=10,
        )
        await _ensure_schema(_pool)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_pool() on startup")
    return _pool


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                id SMALLINT PRIMARY KEY DEFAULT 1,
                elo_ratings JSONB NOT NULL,
                player_stats JSONB NOT NULL,
                h2h_tracker JSONB NOT NULL,
                player_list JSONB NOT NULL,
                trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT single_row CHECK (id = 1)
            );
            """
        )


async def save_artifacts(
    elo_ratings: dict, player_stats: dict, h2h_tracker: dict, player_list: list
) -> None:
    """Upsert the freshly trained state as a single-row snapshot."""
    pool = get_pool()
    # h2h_tracker keys are tuples (player_a, player_b) — JSON needs string keys
    h2h_serializable = {f"{a}||{b}": v for (a, b), v in h2h_tracker.items()}

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_artifacts (id, elo_ratings, player_stats, h2h_tracker, player_list, trained_at)
            VALUES (1, $1, $2, $3, $4, now())
            ON CONFLICT (id) DO UPDATE SET
                elo_ratings = EXCLUDED.elo_ratings,
                player_stats = EXCLUDED.player_stats,
                h2h_tracker = EXCLUDED.h2h_tracker,
                player_list = EXCLUDED.player_list,
                trained_at = EXCLUDED.trained_at;
            """,
            json.dumps(elo_ratings),
            json.dumps(player_stats),
            json.dumps(h2h_serializable),
            json.dumps(player_list),
        )


async def load_artifacts(max_age_hours: int = 24) -> dict | None:
    """
    Return cached artifacts if present and fresh enough, else None.
    ATP data doesn't change intraday, so a day-old snapshot is fine —
    tune max_age_hours to how often you want to pick up new match data.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT elo_ratings, player_stats, h2h_tracker, player_list, trained_at
            FROM model_artifacts
            WHERE id = 1 AND trained_at > now() - ($1 || ' hours')::interval;
            """,
            str(max_age_hours),
        )
    if row is None:
        return None

    h2h_raw = json.loads(row["h2h_tracker"])
    h2h_tracker = {tuple(k.split("||")): v for k, v in h2h_raw.items()}

    return {
        "elo_ratings": json.loads(row["elo_ratings"]),
        "player_stats": json.loads(row["player_stats"]),
        "h2h_tracker": h2h_tracker,
        "player_list": json.loads(row["player_list"]),
    }
