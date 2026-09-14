"""
RoleFit feature engineering (Phase 2).

- Per-90 normalisation of raw counting stats from gold_player_season.
- Possession-adjusted (padj) rates for defensive/creative actions, using
  an event-count share as a possession-time proxy (StatsBomb open data
  doesn't ship a direct time-of-possession field; this is a documented
  v1 simplification, not literal ball-time).
- Correlation check across the per-90 feature set to flag redundant
  clusters (|r| > 0.8) before they get treated as independent signals
  downstream (PCA, similarity, model features).

Writes `gold_player_features` to the same DuckDB file the pipeline built.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
MIN_MINUTES_FOR_CORR = 450  # ~5 full matches, keeps the correlation check from being noise-dominated

RATE_COLS = [
    "goals", "assists", "shot_assists", "total_xg", "shots",
    "passes", "passes_completed", "crosses", "through_balls",
    "dribbles", "dribbles_completed",
    "interceptions", "ball_recoveries", "tackles", "tackles_won",
    "blocks", "clearances", "fouls_committed", "fouls_won",
    "miscontrols", "dispossessed",
]
DEFENSIVE_COLS = ["interceptions", "ball_recoveries", "tackles", "tackles_won", "blocks", "clearances"]


def compute_team_possession_proxy(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Event-count share per team per match, as a possession-time proxy."""
    on_ball_types = ("Pass", "Carry", "Dribble", "Shot", "Ball Receipt*", "Pressure")
    df = con.execute(f"""
        SELECT match_id, possession_team AS team_name, COUNT(*) AS n_events
        FROM bronze_events
        WHERE type IN {on_ball_types} AND possession_team IS NOT NULL
        GROUP BY match_id, possession_team
    """).df()
    totals = df.groupby("match_id")["n_events"].transform("sum")
    df["poss_pct"] = 100 * df["n_events"] / totals
    return df


def build():
    con = duckdb.connect(str(DB_PATH))

    season = con.execute("SELECT * FROM gold_player_season").df()
    count_cols = [c for c in RATE_COLS if c != "total_xg"]
    season[count_cols] = season[count_cols].fillna(0).astype(float)

    # --- per-90 rates ---
    p90 = season[["player_id", "player_name", "primary_position", "matches_played", "total_minutes"]].copy()
    for col in RATE_COLS:
        p90[f"{col}_p90"] = np.where(
            season["total_minutes"] > 0,
            season[col].fillna(0) * 90 / season["total_minutes"],
            np.nan,
        )
    p90["pass_completion_pct"] = np.where(
        season["passes"] > 0, 100 * season["passes_completed"] / season["passes"], np.nan
    )

    # --- possession-adjusted defensive rates ---
    poss = compute_team_possession_proxy(con)
    # map player -> team per match, weight by minutes played that match
    player_team_match = con.execute("""
        SELECT player_id, match_id, team_name, minutes_played
        FROM silver_player_minutes
    """).df()
    merged = player_team_match.merge(poss[["match_id", "team_name", "poss_pct"]], on=["match_id", "team_name"], how="left")
    merged["opp_poss_pct"] = 100 - merged["poss_pct"]

    league_avg_opp_poss = np.average(merged["opp_poss_pct"].dropna(), weights=merged.loc[merged["opp_poss_pct"].notna(), "minutes_played"])

    player_avg_opp_poss = (
        merged.dropna(subset=["opp_poss_pct"])
        .groupby("player_id")
        .apply(lambda g: np.average(g["opp_poss_pct"], weights=g["minutes_played"]), include_groups=False)
        .rename("player_avg_opp_poss_pct")
        .reset_index()
    )

    p90 = p90.merge(player_avg_opp_poss, on="player_id", how="left")
    padj_factor = league_avg_opp_poss / p90["player_avg_opp_poss_pct"]
    for col in DEFENSIVE_COLS:
        p90[f"{col}_padj_p90"] = p90[f"{col}_p90"] * padj_factor

    con.execute("CREATE OR REPLACE TABLE gold_player_features AS SELECT * FROM p90")

    # --- correlation check ---
    feature_cols = [c for c in p90.columns if c.endswith("_p90") or c == "pass_completion_pct"]
    corr_input = p90[p90["total_minutes"] >= MIN_MINUTES_FOR_CORR][feature_cols].dropna(axis=1, how="all")
    corr = corr_input.corr()

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.8:
                pairs.append((cols[i], cols[j], round(r, 3)))
    pairs.sort(key=lambda x: -abs(x[2]))

    con.close()

    print(f"Players in gold_player_features: {len(p90)}")
    print(f"League avg opponent possession proxy: {league_avg_opp_poss:.1f}%")
    print(f"\nFeatures used in correlation check (min {MIN_MINUTES_FOR_CORR} min, n={len(corr_input)}): {len(feature_cols)}")
    print(f"\nHighly correlated pairs (|r| > 0.8):")
    for a, b, r in pairs:
        print(f"  {a:30s} <-> {b:30s}  r={r}")
    if not pairs:
        print("  (none)")

    return p90, pairs


if __name__ == "__main__":
    build()
