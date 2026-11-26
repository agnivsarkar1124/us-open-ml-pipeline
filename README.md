# 🎾 2026 US Open AI Match Analytics & Pre-Draw Engine

An end-to-end ML pipeline and web app that predicts ATP hard-court matchups and explains the factors behind each prediction.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)

**[Live Demo →](https://agnivsarkar1124.github.io/us-open-ml-pipeline/)**

---

## Overview

Pick any two ATP players and the engine returns a calibrated win probability, plus a plain-English breakdown of why.

It replays 2020–2023 ATP results to build a sequential **Elo rating** for every player, then trains a **HistGradientBoostingClassifier** on US Open matches using Elo, head-to-head and tour-experience differentials.

### Highlights
* **Calibrated predictions:** L2-regularized gradient boosting, with probabilities clipped to a realistic 12%–88% band.
* **Explainable output:** every prediction lists its deciding factors (Elo gap, head-to-head record, match experience).
* **Redis caching:** order-independent cache keys with async connection pooling; **~76% lower server-side latency** on repeat matchups (1.05 ms vs 4.4 ms).
* **PostgreSQL persistence:** trained artifacts are stored in Postgres through a pooled async connection layer (psycopg3).
* **Fault-tolerant backend:** the API keeps serving predictions if Redis or Postgres goes down.

---

## Architecture

```text
   GitHub Pages frontend (HTML / CSS / JS)
                  │  REST
                  ▼
   FastAPI backend on Render ── Elo + H2H features → HistGradientBoosting
          │                         │
          ▼                         ▼
   Redis (response cache)    PostgreSQL (trained artifacts)
```

---

## Model Features

Each matchup is scored on three Player A − Player B differentials:

| Feature | Description |
|---|---|
| `elo_diff` | Sequential Elo rating (K = 32) replayed over every match in date order |
| `exp_diff` | Total tour matches played, as a proxy for experience |
| `h2h_diff` | Head-to-head wins between the two players |

Training rows come from US Open matches, with winner/loser order randomized to remove position bias.

---

## API

**Base URL:** `https://two026-us-open-ml-predictor.onrender.com`

| Endpoint | Description |
|---|---|
| `GET /players` | All players in the dataset |
| `POST /predict` | Win probability and breakdown for a matchup |
| `GET /system-status` | Live cache hit/miss stats and database status |

```json
// POST /predict
{ "pA_name": "Novak Djokovic", "pB_name": "Carlos Alcaraz" }
```

```json
{
  "winner": "Novak Djokovic",
  "confidence": 64.2,
  "cached": false,
  "response_time_ms": 4.4,
  "matchup_breakdown": {
    "pA_elo": 2105.3,
    "pB_elo": 1992.8,
    "pA_h2h_wins": 3,
    "pB_h2h_wins": 2,
    "deciding_factors": [
      "Novak Djokovic holds a higher overall Elo rating (2105 vs 1993)",
      "Novak Djokovic leads the Head-to-Head series (3-2)"
    ]
  }
}
```

---

## Run Locally

```bash
git clone https://github.com/agnivsarkar1124/us-open-ml-pipeline
cd us-open-ml-pipeline
pip install -r requirements.txt

# optional: local Postgres + Redis
docker compose up -d
export DATABASE_URL=postgresql://usopen:usopen@localhost:5432/usopen
export REDIS_URL=redis://localhost:6379/0

uvicorn main:app --reload   # http://127.0.0.1:8000/docs
```

Then open `index.html` in a browser.

---

## Roadmap

- [x] 1v1 matchup simulator with explainable predictions
- [x] Probability calibration
- [x] Redis caching + PostgreSQL persistence
- [ ] Hard-court-only historical filtering
- [ ] 128-player bracket simulator for the official 2026 draw

---

MIT License
