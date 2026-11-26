# 🎾 2026 US Open AI Match Analytics & Pre-Draw Engine

An end-to-end ML pipeline and web app that predicts ATP hard-court matchups, backtested on three US Opens it never saw during training.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)

**[Live Demo →](https://agnivsarkar1124.github.io/us-open-ml-pipeline/)**

---

## Overview

Pick any two ATP players and the engine returns a win probability, plus a plain-English breakdown of why.

Under the hood, it replays 16 seasons of ATP results (2010–2025, about 45k matches) to build leakage-free player features. It then trains an ensemble of logistic regression and gradient boosting to predict best-of-five hard-court matches.

### Highlights
* **Rigorously evaluated:** a rolling backtest on the 2023–2025 US Opens (381 held-out matches) with bootstrap confidence intervals, against Elo baselines.
* **Beats Elo:** 69.3% accuracy and a 0.189 Brier score at the US Open, versus 67.7% and 0.194 for Elo. Across 4,781 hard-court matches, the Brier improvement is statistically significant.
* **Leakage-free features:** every feature is computed only from matches played before that match.
* **Redis caching:** order-independent cache keys with async pooling, **0.19 ms cached vs 8.4 ms uncached** median server time (measured locally).
* **PostgreSQL model snapshots:** warm starts load the trained model in about 0.1 s instead of retraining (about 11 s).
* **Fault-tolerant:** the API keeps serving if Redis or Postgres goes down.

---

## Results

Rolling-origin backtest: for each year, train on all main-tour matches before that year's US Open, then predict every main-draw match. Model settings were chosen on the 2021–2022 US Opens before the 2023–2025 results were looked at.

| US Open 2023–2025 (381 matches) | Accuracy | Log loss ↓ | Brier ↓ | AUC |
|---|---|---|---|---|
| **Ensemble (this model)** | **69.3%** | **0.554** | **0.189** | **0.784** |
| Elo rating (K = 32) | 67.7% | 0.563 | 0.194 | 0.773 |
| Elo rating (dynamic K) | 65.4% | 0.574 | 0.198 | 0.767 |
| v1 model (gradient boosting, 3 features) | 63.5% | 0.679 | 0.236 | 0.720 |

* **Larger test set:** across all 4,781 main-tour hard-court matches in 2023–2025, the ensemble beats Elo on Brier score (0.214 vs 0.218, 95% CI of the difference −0.007 to −0.003) and log loss (0.614 vs 0.626).
* **What drives the gain:** an ablation showed that serve and return points won carry most of the signal beyond Elo. Head-to-head and age added nothing.
* **v1 lesson:** the original model, gradient boosting trained only on about 500 US Open matches, lost to plain Elo. Moving to richer features and the full tour dataset fixed it.

<p align="center"><img src="eval/calibration.png" width="420" alt="Calibration plot"></p>

Reproduce with `python evaluate.py`. Full metrics are in [`eval/results.json`](eval/results.json).

---

## Features

All features are Player A minus Player B, computed from matches before the one being predicted:

| Feature | Description |
|---|---|
| Elo | Dynamic-K Elo (K shrinks as a player's match count grows) |
| Hard-court Elo | Separate Elo using only hard-court matches |
| Serve / return | Exponentially weighted % of serve and return points won |
| Form | Elo change over the last 3 months; win rate over the last 10 matches |
| Ranking | Log ATP rank and ranking points |
| Context | Rest days, minutes already played in the tournament, experience, age, head-to-head |

Training data is augmented with both player orders, and predictions average both orders, so output never depends on which player is listed first.

---

## Architecture

```text
   GitHub Pages frontend (HTML / CSS / JS)
                  │  REST
                  ▼
   FastAPI on Render ── features.py → model.py (LogReg + GBM ensemble)
          │                         │
          ▼                         ▼
   Redis (response cache)    PostgreSQL (trained model snapshot)
```

| File | Role |
|---|---|
| `features.py` | Chronological, leakage-free feature builder |
| `model.py` | Ensemble model, shared by training, serving and evaluation |
| `evaluate.py` | Backtest, baselines, bootstrap CIs, calibration plot |
| `main.py` | FastAPI app |
| `cache.py` / `db.py` | Redis cache and Postgres snapshot layers |

---

## API

**Base URL:** `https://two026-us-open-ml-predictor.onrender.com`

| Endpoint | Description |
|---|---|
| `GET /players` | Active players with enough match history |
| `POST /predict` | Win probability and breakdown for a matchup |
| `GET /system-status` | Cache stats, database status, data freshness |

```json
// POST /predict
{ "pA_name": "Jannik Sinner", "pB_name": "Carlos Alcaraz" }
```

```json
{
  "winner": "Jannik Sinner",
  "confidence": 55.0,
  "cached": false,
  "matchup_breakdown": {
    "pA_elo": 2336.6,
    "pB_elo": 2280.3,
    "pA_hard_elo": 2327.1,
    "pB_hard_elo": 2171.1,
    "deciding_factors": [
      "Jannik Sinner is rated higher on hard courts (2327 vs 2171)",
      "Jannik Sinner wins more service points recently (73.7% vs 69.3%)"
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

uvicorn main:app --reload        # http://127.0.0.1:8000/docs

pip install -r requirements-dev.txt
python evaluate.py               # reproduce the backtest
```

---

## Roadmap

- [x] Explainable 1v1 predictions
- [x] Leakage-free backtest against Elo baselines
- [x] Serve/return, surface and form features
- [x] Redis caching + PostgreSQL model snapshots
- [ ] 128-player bracket simulator for the official 2026 draw

---

MIT License
