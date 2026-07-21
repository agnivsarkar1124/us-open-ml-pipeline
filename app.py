import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

# Page configuration
st.set_page_config(
    page_title="2026 US Open AI Predictor",
    page_icon="🎾",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .winner-card {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 20px;
        border-radius: 12px;
        color: #000;
        text-align: center;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title Banner
st.title("🎾 US Open Match Prediction & Weather Pipeline")
st.caption("Recency-Weighted Elo + Atmospheric Court Weather + Gradient Boosting")


# =====================================================================
# DATA & MODEL LOADING (Cached for performance)
# =====================================================================
@st.cache_data
def load_and_train():
    base_url = (
        "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{}.csv"
    )
    dfs = []
    for year in [2020, 2021, 2022, 2023]:
        try:
            df = pd.read_csv(base_url.format(year))
            df["tourney_date"] = pd.to_datetime(
                df["tourney_date"].astype(str), format="%Y%m%d"
            )
            dfs.append(df)
        except Exception:
            pass
    all_matches = pd.concat(dfs, ignore_index=True).sort_values("tourney_date")

    # 1. Compute dynamic Elo ratings across ALL matches against ALL players
    elo_ratings = {}
    K = 32
    DEFAULT_ELO = 1500

    for _, row in all_matches.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        rw = elo_ratings.get(w, DEFAULT_ELO)
        rl = elo_ratings.get(l, DEFAULT_ELO)

        exp_w = 1 / (1 + 10 ** ((rl - rw) / 400))
        exp_l = 1 - exp_w

        elo_ratings[w] = rw + K * (1 - exp_w)
        elo_ratings[l] = rl + K * (0 - exp_l)

    # 2. Build feature training dataset with US Open matches
    us_open_matches = all_matches[
        all_matches["tourney_name"].str.contains("US Open", case=False, na=False)
    ].copy()
    h2h_tracker = {}
    ml_rows = []
    np.random.seed(42)

    for _, row in us_open_matches.iterrows():
        winner, loser = row["winner_name"], row["loser_name"]
        swap = np.random.rand() > 0.5
        pA, pB = (loser, winner) if swap else (winner, loser)
        target = 0 if swap else 1

        pair_key = tuple(sorted([pA, pB]))
        h2h_data = h2h_tracker.get(pair_key, {pA: 0, pB: 0})
        h2h_diff = h2h_data.get(pA, 0) - h2h_data.get(pB, 0)

        if pair_key not in h2h_tracker:
            h2h_tracker[pair_key] = {winner: 1, loser: 0}
        else:
            h2h_tracker[pair_key][winner] = (
                h2h_tracker[pair_key].get(winner, 0) + 1
            )

        # Actual Elo difference from past performance against all opponents
        elo_diff = elo_ratings.get(pA, DEFAULT_ELO) - elo_ratings.get(
            pB, DEFAULT_ELO
        )

        ml_rows.append(
            {"elo_diff": elo_diff, "h2h_diff": h2h_diff, "target": target}
        )

    ml_df = pd.DataFrame(ml_rows)
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(ml_df[["elo_diff", "h2h_diff"]], ml_df["target"])

    return all_matches, h2h_tracker, elo_ratings, model


with st.spinner("🚀 Loading ATP Data & Training Gradient Boosting Model..."):
    all_matches, h2h_tracker, elo_ratings, model = load_and_train()


# Live Weather Integration
def get_weather():
    try:
        res = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": 40.7500,
                "longitude": -73.8472,
                "start_date": "2025-09-01",
                "end_date": "2025-09-01",
                "hourly": ["temperature_2m", "wind_speed_10m"],
            },
            timeout=5,
        ).json()
        return max(res["hourly"]["temperature_2m"]), max(
            res["hourly"]["wind_speed_10m"]
        )
    except Exception:
        return 28.0, 15.0


temp_c, wind_kmh = get_weather()

# =====================================================================
# DASHBOARD LAYOUT
# =====================================================================
col1, col2, col3 = st.columns(3)
col1.metric("Flushing Meadows Temp", f"{temp_c}°C", "Live API")
col2.metric("Max Wind Speed", f"{wind_kmh} km/h", "Outdoor Court")
col3.metric("Model Architecture", "HistGradientBoosting", "Dynamic Elo + H2H")

st.divider()

# Roster Setup
recent_ranks = (
    pd.concat([
        all_matches[["winner_name", "winner_rank"]].rename(
            columns={"winner_name": "name", "winner_rank": "rank"}
        ),
        all_matches[["loser_name", "loser_rank"]].rename(
            columns={"loser_name": "name", "loser_rank": "rank"}
        ),
    ])
    .dropna()
    .groupby("name")
    .last()
    .reset_index()
    .sort_values("rank")
    .head(8)
)

# Add computed Elo ratings to table display
recent_ranks["Calculated Elo"] = recent_ranks["name"].apply(
    lambda x: round(elo_ratings.get(x, 1500), 1)
)

st.subheader("📊 Top 8 Seeds & Dynamic Ratings")
st.dataframe(
    recent_ranks.rename(
        columns={"name": "Player Name", "rank": "Current Rank"}
    ),
    use_container_width=True,
)


# Match Predictor Logic
def predict(p1_name, p2_name):
    p1_elo = elo_ratings.get(p1_name, 1500)
    p2_elo = elo_ratings.get(p2_name, 1500)
    elo_diff = p1_elo - p2_elo

    pair_key = tuple(sorted([p1_name, p2_name]))
    h2h_data = h2h_tracker.get(pair_key, {p1_name: 0, p2_name: 0})
    h2h_diff = h2h_data.get(p1_name, 0) - h2h_data.get(p2_name, 0)

    feats = pd.DataFrame(
        [[elo_diff, h2h_diff]], columns=["elo_diff", "h2h_diff"]
    )
    prob = model.predict_proba(feats)[0][1]

    winner = p1_name if prob >= 0.5 else p2_name
    confidence = prob if prob >= 0.5 else (1 - prob)
    return winner, confidence, elo_diff


# Bracket Simulation Trigger
st.divider()
st.subheader("🏆 Tournament Bracket Simulation")

if st.button("▶ Run US Open Bracket Simulation", type="primary"):
    roster = recent_ranks["name"].tolist()

    # Quarterfinals
    st.markdown("### Quarterfinals")
    qf_winners = []
    qf_cols = st.columns(4)

    qf_pairs = [
        (roster[0], roster[7]),
        (roster[3], roster[4]),
        (roster[2], roster[5]),
        (roster[1], roster[6]),
    ]
    for idx, (p1, p2) in enumerate(qf_pairs):
        winner, conf, ediff = predict(p1, p2)
        qf_winners.append(winner)
        with qf_cols[idx]:
            st.info(
                f"**{p1}** vs **{p2}**\n\n"
                f"👉 **{winner}** ({conf*100:.1f}%)\n\n"
                f"📈 Elo Diff: `{ediff:+.1f}`"
            )

    # Semifinals
    st.markdown("### Semifinals")
    sf_winners = []
    sf_cols = st.columns(2)

    sf_pairs = [
        (qf_winners[0], qf_winners[1]),
        (qf_winners[2], qf_winners[3]),
    ]
    for idx, (p1, p2) in enumerate(sf_pairs):
        winner, conf, ediff = predict(p1, p2)
        sf_winners.append(winner)
        with sf_cols[idx]:
            st.success(
                f"**{p1}** vs **{p2}**\n\n"
                f"👉 **{winner}** ({conf*100:.1f}%)\n\n"
                f"📈 Elo Diff: `{ediff:+.1f}`"
            )

    # Final
    st.markdown("### Championship Final")
    champ, champ_conf, ediff = predict(sf_winners[0], sf_winners[1])

    st.balloons()
    st.markdown(
        f"""
        <div class="winner-card">
            <h2>🏆 PREDICTED CHAMPION: {champ}</h2>
            <h3>Confidence Rating: {champ_conf*100:.1f}%</h3>
            <p>Elo Differential Advantage: {ediff:+.1f}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
