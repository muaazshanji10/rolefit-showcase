"""
RoleFit: Premier League 2015/16 recruitment analytics showcase.

Reads precomputed parquet snapshots (app/data/*.parquet) exported from
the DuckDB gold tables by src/export_app_data.py -- the app itself
never touches raw StatsBomb data or re-runs the pipeline, so it works
standalone on Streamlit Community Cloud.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# -- palette (dataviz skill reference palette, light mode) --
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

FEATURES = [
    "total_xg_p90", "shots_p90", "assists_p90", "shot_assists_p90",
    "crosses_p90", "through_balls_p90",
    "passes_completed_p90", "pass_completion_pct",
    "dribbles_completed_p90", "dispossessed_p90",
    "interceptions_padj_p90", "ball_recoveries_padj_p90",
    "tackles_won_padj_p90", "blocks_padj_p90", "clearances_padj_p90",
    "fouls_committed_p90", "fouls_won_p90",
]
FEATURE_DESCRIPTIONS = {
    "total_xg_p90": "expected goals generated per 90 minutes (shot quality x volume)",
    "shots_p90": "shots taken per 90 minutes",
    "assists_p90": "assists per 90 minutes",
    "shot_assists_p90": "passes leading directly to a shot, per 90 minutes (chance creation)",
    "crosses_p90": "crosses played per 90 minutes",
    "through_balls_p90": "defense-splitting through balls per 90 minutes",
    "passes_completed_p90": "completed passes per 90 minutes (passing volume)",
    "pass_completion_pct": "% of attempted passes completed (passing reliability)",
    "dribbles_completed_p90": "successful dribbles per 90 minutes (beating a defender 1v1)",
    "dispossessed_p90": "times dispossessed of the ball per 90 minutes (ball security, lower is better)",
    "interceptions_padj_p90": "interceptions per 90 minutes, possession-adjusted",
    "ball_recoveries_padj_p90": "loose-ball recoveries per 90 minutes, possession-adjusted",
    "tackles_won_padj_p90": "tackles won per 90 minutes, possession-adjusted",
    "blocks_padj_p90": "shots/passes blocked per 90 minutes, possession-adjusted",
    "clearances_padj_p90": "defensive clearances per 90 minutes, possession-adjusted",
    "fouls_committed_p90": "fouls committed per 90 minutes",
    "fouls_won_p90": "fouls won per 90 minutes",
}
SIMILARITY_MIN_MINUTES = 450

st.set_page_config(page_title="RoleFit", page_icon="⚽", layout="wide")


@st.cache_data
def load_data():
    features = pd.read_parquet(DATA_DIR / "gold_player_features.parquet")
    roles = pd.read_parquet(DATA_DIR / "gold_player_roles.parquet")
    shrinkage = pd.read_parquet(DATA_DIR / "gold_goals_shrinkage.parquet")
    return features, roles, shrinkage


@st.cache_data
def cluster_centroids(_roles: pd.DataFrame):
    X = _roles[FEATURES].fillna(0)
    Xz = StandardScaler().fit_transform(X)
    Xz_df = pd.DataFrame(Xz, columns=FEATURES, index=_roles.index)
    Xz_df["cluster"] = _roles["cluster"].values
    Xz_df["cluster_name"] = _roles["cluster_name"].values
    return Xz_df.groupby(["cluster", "cluster_name"])[FEATURES].mean()


def similarity_pool(features: pd.DataFrame):
    pool = features[features["total_minutes"] >= SIMILARITY_MIN_MINUTES].reset_index(drop=True)
    Xz = StandardScaler().fit_transform(pool[FEATURES].fillna(0))
    return pool, Xz


def styled_barh(labels, values, colors=None, xlabel="", title="", figsize=(6, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = [BLUE if v >= 0 else RED for v in values]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, colors=TEXT_SECONDARY)
    ax.axvline(0, color="#c9c8c1", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#c9c8c1")
    fig.tight_layout()
    return fig


st.title("⚽ RoleFit")
st.caption("Recruitment analytics on Premier League 2015/16 (StatsBomb open data) — v1 showcase, single season, light tuning.")

features, roles, shrinkage = load_data()
centroids = cluster_centroids(roles)

tab_search, tab_role, tab_sim, tab_nl, tab_proj = st.tabs(
    ["🔍 Search", "🎯 Role Fit", "👥 Similar Players", "💬 NL Brief", "📊 Projections"]
)

with tab_search:
    st.subheader("Player search")
    st.markdown("Look up any player from the 2015/16 Premier League season and see their basic profile at a glance — position, games played, and a few headline stats. A good starting point if you already have a name in mind.")
    query = st.text_input("Player name contains", "")
    df = features.copy()
    if query:
        df = df[df["player_name"].str.contains(query, case=False, na=False)]
    df = df.merge(roles[["player_id", "cluster_name"]], on="player_id", how="left")
    df["cluster_name"] = df["cluster_name"].fillna("(< 900 min, not clustered)")
    show_cols = ["player_name", "primary_position", "cluster_name", "matches_played", "total_minutes"] + [
        c for c in ["total_xg_p90", "assists_p90", "tackles_won_padj_p90", "pass_completion_pct"] if c in df.columns
    ]
    st.dataframe(
        df.sort_values("total_minutes", ascending=False)[show_cols].round(2),
        use_container_width=True, height=450,
    )

with tab_role:
    st.subheader("Role fit: how closely does a player match each hand-built role?")
    st.markdown("Players naturally fall into groups based on how they actually play — not just their listed position. This shows which of those playing-style groups (e.g. \"ball-winning midfielder\" or \"advanced forward\") a player best matches, useful for spotting players who fit a tactical role even if their official position label doesn't obviously say so.")
    player = st.selectbox("Player", sorted(roles["player_name"].unique()), key="role_player")
    row = roles[roles["player_name"] == player].iloc[0]
    X = row[FEATURES].fillna(0).to_numpy().astype(float)
    scaler = StandardScaler().fit(roles[FEATURES].fillna(0))
    xz = scaler.transform([X])[0]

    sims = {}
    for (cid, cname), crow in centroids.iterrows():
        c = crow.to_numpy()
        sims[cname] = float(np.dot(xz, c) / (np.linalg.norm(xz) * np.linalg.norm(c) + 1e-9))
    sims = dict(sorted(sims.items(), key=lambda x: -x[1]))

    st.markdown(f"**{player}** — primary position: {row['primary_position']} · assigned cluster: **{row['cluster_name']}**")
    fig = styled_barh(list(sims.keys()), list(sims.values()), colors=CATEGORICAL[:len(sims)],
                       xlabel="cosine similarity to role centroid", title="Role fit")
    st.pyplot(fig)

with tab_sim:
    st.subheader("Find players like X")
    st.markdown("Pick a player and get a shortlist of others with a similar statistical profile — handy for finding a cheaper or more available alternative to a player you like, or simply for comparing options.")
    pool, Xz = similarity_pool(features)
    player = st.selectbox("Player", sorted(pool["player_name"].unique()), key="sim_player")
    n = st.slider("Number of results", 3, 15, 5)
    idx = pool.index[pool["player_name"] == player][0]

    euc = euclidean_distances(Xz[[idx]], Xz)[0]
    cos = cosine_similarity(Xz[[idx]], Xz)[0]
    euc_order = [i for i in np.argsort(euc) if i != idx][:n]
    cos_order = [i for i in np.argsort(-cos) if i != idx][:n]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Euclidean distance** (lower = more similar)")
        t = pool.iloc[euc_order][["player_name", "primary_position", "total_minutes"]].copy()
        t["distance"] = euc[euc_order]
        st.dataframe(t.round(2), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**By cosine similarity** (higher = more similar)")
        t = pool.iloc[cos_order][["player_name", "primary_position", "total_minutes"]].copy()
        t["similarity"] = cos[cos_order]
        st.dataframe(t.round(3), use_container_width=True, hide_index=True)
    st.caption(f"Pool: {len(pool)} players with ≥{SIMILARITY_MIN_MINUTES} minutes.")

with tab_nl:
    st.subheader("Describe the player you want")
    st.markdown("No need to know the stats — just describe what you're after in plain English (e.g. \"a tough tackling defensive midfielder\") and this turns that description into a ranked list of real players who fit it best.")
    st.caption("A real Anthropic API call converts your brief into a weight over the same 17 features used for role discovery, then ranks every eligible player by that weighted profile.")
    brief = st.text_area("Brief", "A tough ball-winning defensive midfielder who keeps possession simple.", height=80)
    if st.button("Generate weights & rank players"):
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
        if not api_key:
            st.error("No ANTHROPIC_API_KEY found in st.secrets or environment.")
        else:
            with st.spinner("Calling Claude..."):
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                system_prompt = (
                    "You are a football (soccer) recruitment analyst. You convert a plain-English "
                    "scouting brief into a structured weight vector over a fixed set of per-90-minute "
                    "statistical features, so the weights can be used to numerically score and rank real "
                    "players.\n\nAvailable features (name: meaning):\n"
                    + "\n".join(f"- {k}: {v}" for k, v in FEATURE_DESCRIPTIONS.items())
                    + "\n\nReturn ONLY a JSON object mapping every one of the above feature names to a "
                    "weight in [-1, 1]. Positive = more is better; negative = less is better; near zero "
                    "= not relevant. Output strictly valid JSON, no prose, no markdown fences."
                )
                resp = client.messages.create(
                    model="claude-sonnet-5", max_tokens=1024, system=system_prompt,
                    messages=[{"role": "user", "content": f"Brief: {brief}"}],
                )
                text = next(b.text for b in resp.content if b.type == "text").strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    text = text[text.find("{"):text.rfind("}") + 1]
                weights = json.loads(text)
                weights = {k: float(v) for k, v in weights.items() if k in FEATURES}

            top_w = sorted(weights.items(), key=lambda x: -abs(x[1]))[:8]
            fig = styled_barh([k for k, v in top_w], [v for k, v in top_w],
                               xlabel="weight", title="LLM-derived feature weights (top 8)")
            st.pyplot(fig)

            pool, Xz = similarity_pool(features)
            w = np.array([weights.get(f, 0.0) for f in FEATURES])
            scores = Xz @ w
            pool = pool.copy()
            pool["brief_score"] = scores
            top_players = pool.sort_values("brief_score", ascending=False).head(10)
            st.markdown("**Top 10 players ranked by this brief**")
            st.dataframe(
                top_players[["player_name", "primary_position", "total_minutes", "brief_score"]].round(2),
                use_container_width=True, hide_index=True,
            )

            w_norm = w / (np.linalg.norm(w) + 1e-9)
            sims = {}
            for (cid, cname), crow in centroids.iterrows():
                c = crow.to_numpy()
                sims[cname] = float(np.dot(w_norm, c / (np.linalg.norm(c) + 1e-9)))
            best_cluster = max(sims, key=sims.get)
            st.caption(f"Closest hand-built role: **{best_cluster}** (cosine sim {sims[best_cluster]:.2f})")

with tab_proj:
    st.subheader("Goals/90: raw vs shrunk, with uncertainty")
    st.markdown("A player's raw scoring rate can be misleading if they've only played a handful of games — a couple of lucky goals can make someone look like a world-beater. This shows a more realistic estimate alongside the raw number, plus a range showing how confident we actually are in it, so you can tell a proven scorer from a small-sample fluke.")
    st.caption("Empirical-Bayes shrinkage pulls small-sample players toward the population mean; error bars are bootstrap 95% CIs on the raw rate.")
    min_min = st.slider("Minimum minutes played", 90, 3000, 900, step=90)
    d = shrinkage[shrinkage["total_minutes"] >= min_min].sort_values("shrunk_p90", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(d))
    err_low = (d["raw_p90"] - d["boot_ci_low"]).clip(lower=0)
    err_high = (d["boot_ci_high"] - d["raw_p90"]).clip(lower=0)
    ax.errorbar(d["raw_p90"], y_pos, xerr=[err_low, err_high], fmt="o", color=TEXT_SECONDARY,
                ecolor="#c9c8c1", elinewidth=1.5, capsize=3, markersize=5, label="raw (95% bootstrap CI)")
    ax.scatter(d["shrunk_p90"], y_pos, color=BLUE, s=45, zorder=5, label="shrunk (empirical Bayes)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["player_name"], fontsize=9, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlabel("goals per 90", color=TEXT_SECONDARY, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#c9c8c1")
    ax.tick_params(left=False, colors=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        d[["player_name", "matches_played", "total_minutes", "goals", "raw_p90", "boot_ci_low", "boot_ci_high", "shrunk_p90"]].round(3),
        use_container_width=True, hide_index=True,
    )

st.divider()
col1, col2 = st.columns([1, 4])
with col1:
    st.image(str(APP_DIR / "assets" / "statsbomb_logo.png"), width=140)
with col2:
    st.caption(
        "Data: StatsBomb open data (Premier League 2015/16). "
        "Free for public, non-commercial use — see [statsbomb.com/what-we-do/hub/free-data/](https://statsbomb.com/what-we-do/hub/free-data/). "
        "This is a v1 showcase: single league/season, light tuning. Roadmap: four-league combination, deeper hyperparameter search."
    )
