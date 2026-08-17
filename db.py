"""
PostgreSQL layer: connection pooling + persistence for trained artifacts
(elo_ratings, player_stats, h2h_tracker) so a cold start can skip
re-downloading and re-training against 4 years of ATP CSVs.

Uses psycopg (v3) with psycopg_pool instead of asyncpg — asyncpg ships no
prebuilt wheel for newer Python versions (e.g. 3.14) and fails to compile
from source on some toolchains. psycopg3 has prebuilt wheels via the
`psycopg[binary]` extra and an equivalent async pooled API.
"""

import json
import os

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]  # postgresql://user:pass@host:port/db

# Tune pool size to your Render/Postgres plan. min_size keeps warm connections
# ready so the first request after idle doesn't pay a connection-setup cost.
_pool: AsyncConnectionPool | None = None


async def init_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=2,
            max_size=10,
            timeout=10,
            open=False,
        )
        await _pool.open(wait=True)
        await _ensure_schema(_pool)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_pool() on startup")
    return _pool


async def _ensure_schema(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
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

    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO model_artifacts (id, elo_ratings, player_stats, h2h_tracker, player_list, trained_at)
            VALUES (1, %s, %s, %s, %s, now())
            ON CONFLICT (id) DO UPDATE SET
                elo_ratings = EXCLUDED.elo_ratings,
                player_stats = EXCLUDED.player_stats,
                h2h_tracker = EXCLUDED.h2h_tracker,
                player_list = EXCLUDED.player_list,
                trained_at = EXCLUDED.trained_at;
            """,
            (
                json.dumps(elo_ratings),
                json.dumps(player_stats),
                json.dumps(h2h_serializable),
                json.dumps(player_list),
            ),
        )


async def load_artifacts(max_age_hours: int = 24) -> dict | None:
    """
    Return cached artifacts if present and fresh enough, else None.
    ATP data doesn't change intraday, so a day-old snapshot is fine —
    tune max_age_hours to how often you want to pick up new match data.
    """
    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT elo_ratings, player_stats, h2h_tracker, player_list, trained_at
                FROM model_artifacts
                WHERE id = 1 AND trained_at > now() - (%s || ' hours')::interval;
                """,
                (str(max_age_hours),),
            )
            row = await cur.fetchone()

    if row is None:
        return None

    # psycopg auto-adapts JSONB columns to Python dicts/lists already,
    # so no json.loads needed here (unlike the raw-driver version).
    h2h_raw = row["h2h_tracker"]
    h2h_tracker = {tuple(k.split("||")): v for k, v in h2h_raw.items()}

    return {
        "elo_ratings": row["elo_ratings"],
        "player_stats": row["player_stats"],
        "h2h_tracker": h2h_tracker,
        "player_list": row["player_list"],
    }
