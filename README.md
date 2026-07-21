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
* **Calibrated Win Probabilities:** Features l2-regularized probability bounds (capped at realistic ~12%–88% sports variance bounds) to eliminate uncalibrated tree extremity.
* **Feature Attribution & "Why This Prediction?":** Breaks down head-to-head metrics, overall win-rate differentials, and match experience to explain *how* the model reached its verdict.
* **Pre-Draw Positioning:** Designed as a flexible scenario simulator ahead of the full 128-player US Open bracket release.

---

## 🛠️ Tech Stack & Architecture
