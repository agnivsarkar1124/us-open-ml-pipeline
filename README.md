# 🎾 2026 US Open AI Match Analytics & Pre-Draw Engine

An end-to-end Machine Learning pipeline and web application built to predict high-stakes ATP hard-court matchups and explain the feature attributions behind every prediction.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient_Boosting-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub_Pages-222222?style=flat&logo=github&logoColor=white)
![Render](https://img.shields.io/badge/Hosting-Render-46E3B7?style=flat&logo=render&logoColor=white)

---

## 📌 Overview

The **2026 US Open AI Match Analytics Engine** is an interactive scouting tool designed to evaluate 1v1 ATP tennis matchups prior to the release of the official tournament draw. 

Using historical ATP match performance datasets (2020–2023), the engine trains a **HistGradientBoostingClassifier** on relative win rates, head-to-head records, and ATP Tour match density to deliver calibrated win probabilities alongside human-interpretable decision factors.

### 🌟 Key Features
* **Interactive 1v1 Matchup Engine:** Select any two players from a dynamically loaded ATP roster to generate real-time match predictions.
* **Calibrated Win Probabilities:** Features L2-regularized probability bounds (capped at realistic ~12%–88% sports variance bounds) to eliminate uncalibrated tree extremity.
* **Feature Attribution & "Why This Prediction?":** Breaks down head-to-head metrics, overall win-rate differentials, and match experience to explain *how* the model reached its verdict.
* **Pre-Draw Positioning:** Designed as a flexible scenario simulator ahead of the full 128-player US Open bracket release.

---

## 🛠️ Tech Stack & Architecture

```text
                   +----------------------------------+
                   |    GitHub Pages (index.html)     |
                   |   Vanilla JS / CSS3 UI Engine    |
                   +----------------------------------+
                                    |
                                    | REST API (POST /predict)
                                    v
                   +----------------------------------+
                   |        FastAPI Backend           |
                   |      (Hosted on Render)          |
                   +----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  ATP Historical Data  |                       | HistGradientBoosting  |
|  (2020-2023 Datasets) |                       |   Classifier Model    |
+-----------------------+                       +-----------------------+
```

* **Frontend:** Single-Page Application (SPA) hosted on **GitHub Pages**, styled with custom dark-mode CSS and asynchronous JavaScript fetch pipelines.
* **Backend:** **FastAPI** application deployed on **Render** with CORS middleware enabled.
* **Machine Learning:** `scikit-learn` (`HistGradientBoostingClassifier`), `pandas`, and `numpy`.

---

## 📊 Feature Pipeline

The model calculates three core features for every matchup:

1. **Win Rate Differential (`win_rate_diff`):** Calculates overall career win rates on the ATP Tour across historical datasets.
2. **Match Density Differential (`exp_diff`):** Measures total professional matches played to account for veteran experience and data density.
3. **Head-to-Head Differential (`h2h_diff`):** Tracks historical match wins between Player A and Player B.

---

## 🚀 API Endpoint Reference

### Base URL
`https://two026-us-open-ml-predictor.onrender.com`

### `GET /players`
Returns the list of all available ATP players parsed from the historical match dataset.

### `POST /predict`
Executes match prediction and feature attribution.

**Request Body:**
```json
{
  "pA_name": "Novak Djokovic",
  "pB_name": "Carlos Alcaraz"
}
```

**Response Example:**
```json
{
  "winner": "Novak Djokovic",
  "confidence": 64.2,
  "elo_diff": 112.5,
  "h2h_diff": 1,
  "matchup_breakdown": {
    "pA_name": "Novak Djokovic",
    "pB_name": "Carlos Alcaraz",
    "pA_h2h_wins": 3,
    "pB_h2h_wins": 2,
    "pA_total_atp_wins": 182,
    "pB_total_atp_wins": 124,
    "deciding_factors": [
      "Novak Djokovic holds a higher overall ATP win rate (83.1%)",
      "Novak Djokovic leads the Head-to-Head series (3-2)",
      "Novak Djokovic has significantly higher ATP Tour match density"
    ]
  }
}
```

---

## 🔧 Local Setup & Installation

### Prerequisites
* Python 3.9+
* Pip

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn
```

### 3. Run the FastAPI backend locally
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

### 4. View the frontend
Simply open `index.html` in your web browser, or serve it using Python's built-in HTTP server:
```bash
python -m http.server 8080
```

---

## 🗓️ Roadmap & Future Enhancements

- [x] Pre-draw 1v1 matchup simulator.
- [x] Feature attribution and "Why this prediction?" breakdown.
- [x] Probability calibration and clipping bounds.
- [ ] Surface-specific (Hard Court only) historical filtering.
- [ ] 128-Player Tournament Bracket Simulator *(Unlocks upon official 2026 US Open Draw reveal)*.

---

## 📜 License

Distributed under the MIT License.
