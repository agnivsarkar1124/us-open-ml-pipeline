# 🎾 US Open Match Prediction & Weather-Adjusted Elo Pipeline

An end-to-end Machine Learning pipeline that predicts US Open tennis match outcomes using dynamic recency-weighted Elo ratings, atmospheric court weather integration, and gradient boosting classifiers.

## 🚀 Key Features
* **Recency-Weighted Elo Engine:** Calculates player ratings with exponential time-decay modeling across 10,000+ ATP match records.
* **Live Weather Integration:** Fetches real-time temperature and wind data via the Open-Meteo API to adjust player Elo parameters based on stamina and ace efficiency.
* **Leak-Free ML Feature Engineering:** Computes pre-match Elo differentials, hard court win percentages, and dynamic Head-to-Head (H2H) records.
* **Tournament Bracket Simulation:** Trains a `HistGradientBoostingClassifier` to simulate multi-round bracket match predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **ML / Stats:** Scikit-Learn (`HistGradientBoostingClassifier`), NumPy, Pandas
* **API Integration:** Open-Meteo API (REST / JSON)

## 🏃 Getting Started

### Prerequisites
```bash
pip install pandas numpy requests scikit-learn
