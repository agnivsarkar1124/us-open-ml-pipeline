"""
Out-of-sample evaluation for the US Open match predictor.

Protocol
--------
* Features come from features.FeatureBuilder: each match's features use only
  matches played before it. 2010–2013 only warm up the ratings.
* Rolling-origin backtest: for each test year Y in 2023–2025, the model is
  trained on main-tour matches before the Y US Open starts, then predicts all
  127 main-draw US Open matches of year Y (381 total).
* Model settings were chosen on US Open 2021–2022 before the 2023–2025
  results were looked at.
* Secondary test: every main-tour hard-court match in 2023–2025, each year
  predicted by a model trained on matches before Jan 1 of that year.
* Baselines: Elo expected score (fixed K=32 and dynamic K), and the v1
  production model (HistGBM on 3 features, trained on US Open matches only).

Usage:
    python evaluate.py                    # downloads CSVs from GitHub
    python evaluate.py --data-dir ./data  # or reads local atp_matches_YYYY.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from features import MAIN_TOUR_LEVELS, FeatureBuilder, is_us_open, load_matches
from model import MatchModel, add_model_columns, elo_expected, training_rows

YEARS = range(2010, 2026)
TEST_YEARS = [2023, 2024, 2025]
V1_FEATURES = ["elo_fixed_diff", "exp_diff", "h2h_diff"]


# ---------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ece": ece(y, p),
    }


def bootstrap_diff(y, pa, pb, fn, n=4000, seed=0):
    """95% CI of fn(model A) - fn(model B), resampling matches."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def acc_fn(y, p):
    return accuracy_score(y, p >= 0.5)


# ---------------------------------------------------------------- models
def v1_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Replica of the original production model (US Open rows only, clipped)."""
    uso = train[train.tourney_name.map(is_us_open) & (train.year >= 2020)]
    m = HistGradientBoostingClassifier(random_state=42, max_iter=100,
                                       min_samples_leaf=10, l2_regularization=1.0)
    m.fit(uso[V1_FEATURES], uso.target)
    return np.clip(m.predict_proba(test[V1_FEATURES])[:, 1], 0.12, 0.88)


def run(data_dir, out_dir: Path) -> dict:
    rows = add_model_columns(
        FeatureBuilder().process(load_matches(YEARS, data_dir), record_from_year=2014)
    )
    uso_mask = rows.tourney_name.map(is_us_open)

    # ---- primary: US Open 2023–2025
    preds = []
    for ty in TEST_YEARS:
        test = rows[uso_mask & (rows.year == ty)].copy()
        train = training_rows(rows, before=test.date.min())
        test["p_model"] = MatchModel().fit(train).predict_proba(test)
        test["p_elo_fixed"] = elo_expected(test.elo_fixed_diff)
        test["p_elo_dyn"] = elo_expected(test.elo_diff)
        test["p_v1"] = v1_model(train, test)
        test["n_train"] = len(train)
        preds.append(test)
    uso = pd.concat(preds, ignore_index=True)
    y = uso.target.to_numpy()

    names = {
        "Ensemble (this model)": "p_model",
        "Elo, fixed K=32": "p_elo_fixed",
        "Elo, dynamic K": "p_elo_dyn",
        "v1 model (HistGBM, 3 features)": "p_v1",
    }
    report = {
        "test_set": f"US Open {TEST_YEARS[0]}–{TEST_YEARS[-1]} main draw",
        "n_test_matches": int(len(uso)),
        "us_open": {k: metrics(y, uso[c]) for k, c in names.items()},
        "by_year": {
            int(yr): {"n_train": int(g.n_train.iloc[0]),
                      **{k: metrics(g.target.to_numpy(), g[c]) for k, c in names.items()}}
            for yr, g in uso.groupby("year")
        },
        "ci_vs_elo_fixed": {
            "brier": bootstrap_diff(y, uso.p_model.to_numpy(), uso.p_elo_fixed.to_numpy(), brier_score_loss),
            "accuracy": bootstrap_diff(y, uso.p_model.to_numpy(), uso.p_elo_fixed.to_numpy(), acc_fn),
        },
        "ci_vs_elo_dynamic": {
            "brier": bootstrap_diff(y, uso.p_model.to_numpy(), uso.p_elo_dyn.to_numpy(), brier_score_loss),
        },
    }

    # ---- secondary: all main-tour hard-court matches 2023–2025
    hp, hy, hf = [], [], []
    for ty in TEST_YEARS:
        test = rows[(rows.year == ty) & rows.tourney_level.isin(MAIN_TOUR_LEVELS) & (rows.surface == "Hard")]
        train = training_rows(rows, before=pd.Timestamp(f"{ty}-01-01"))
        hp.append(MatchModel().fit(train).predict_proba(test))
        hf.append(elo_expected(test.elo_fixed_diff))
        hy.append(test.target.to_numpy())
    hp, hf, hy = map(np.concatenate, (hp, hf, hy))
    report["hard_court"] = {
        "n_test_matches": int(len(hy)),
        "Ensemble (this model)": metrics(hy, hp),
        "Elo, fixed K=32": metrics(hy, hf),
        "ci_vs_elo_fixed": {
            "brier": bootstrap_diff(hy, hp, hf, brier_score_loss),
            "accuracy": bootstrap_diff(hy, hp, hf, acc_fn),
        },
    }

    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    uso[["year", "round", "pA", "pB", "target", "p_model", "p_elo_fixed", "p_elo_dyn", "p_v1"]].to_csv(
        out_dir / "predictions.csv", index=False)
    plot_calibration(uso, out_dir / "calibration.png")
    return report


def plot_calibration(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text, text2, grid = "#191919", "#666666", "#e7e7e7"
    series = [("Ensemble (this model)", "p_model", "#257de0", "o"),
              ("Elo, fixed K=32", "p_elo_fixed", "#f47230", "s"),
              ("v1 model", "p_v1", "#00b894", "D")]
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    ax.plot([0, 1], [0, 1], color="#9c9c9c", lw=1.2, ls="--", label="Perfect calibration", zorder=1)
    y = df.target.to_numpy()
    for label, col, color, marker in series:
        p = df[col].to_numpy()
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        pts = [(p[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 8]
        ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=8,
                markeredgecolor="white", markeredgewidth=2, label=label, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Predicted win probability", color=text2)
    ax.set_ylabel("Observed win rate", color=text2)
    ax.set_title(f"Held-out calibration: US Open {TEST_YEARS[0]}–{TEST_YEARS[-1]}",
                 color=text, fontsize=11, loc="left")
    ax.grid(color=grid, lw=0.8)
    ax.tick_params(colors=text2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=text, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def print_report(r: dict) -> None:
    def table(d):
        print(f"  {'Model':32s} {'Acc':>6s} {'LogLoss':>8s} {'Brier':>7s} {'AUC':>6s} {'ECE':>6s}")
        for k, m in d.items():
            if isinstance(m, dict) and "auc" in m:
                print(f"  {k:32s} {m['accuracy']:6.3f} {m['log_loss']:8.3f} {m['brier']:7.4f} {m['auc']:6.3f} {m['ece']:6.3f}")
    print(f"\n{r['test_set']} — {r['n_test_matches']} held-out matches")
    table(r["us_open"])
    c = r["ci_vs_elo_fixed"]
    print(f"  vs fixed-K Elo, 95% CI: Brier {c['brier'][0]:+.4f}..{c['brier'][1]:+.4f}, "
          f"accuracy {c['accuracy'][0]:+.3f}..{c['accuracy'][1]:+.3f}")
    print(f"\nAll main-tour hard-court matches 2023–2025 — {r['hard_court']['n_test_matches']} matches")
    table(r["hard_court"])
    c = r["hard_court"]["ci_vs_elo_fixed"]
    print(f"  vs fixed-K Elo, 95% CI: Brier {c['brier'][0]:+.4f}..{c['brier'][1]:+.4f}, "
          f"accuracy {c['accuracy'][0]:+.3f}..{c['accuracy'][1]:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    print_report(run(args.data_dir, Path(args.out)))
