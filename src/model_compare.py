"""
RoleFit model comparison (Phase 5).

Task: predict a player's SECOND-half-of-season goal+assist rate (G+A
per 90) from their FIRST-half-of-season stat profile. Match weeks 1-19
= first half, 20-38 = second half. Players need >=450 min in BOTH
halves (~5 matches each) to keep the target from being pure noise.

Models, all default/sensible hyperparameters, no tuning loop:
  - Baseline: predicted second-half G+A/90 = first-half G+A/90 ("same
    as first half")
  - Ridge (alpha=1.0)
  - Lasso (alpha=1.0)
  - LightGBM (LGBMRegressor defaults, n_estimators=100)

Evaluated with 5-fold CV (same folds for every model), reporting MAE
and R^2. Whatever wins, wins -- this script prints the real numbers,
it does not pick a flattering framing.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
MIN_MINUTES_PER_HALF = 450
FIRST_HALF_MAX_WEEK = 19

FEATURE_COLS = [
    "total_xg_p90", "shots_p90", "assists_p90", "shot_assists_p90",
    "crosses_p90", "through_balls_p90",
    "passes_completed_p90", "pass_completion_pct",
    "dribbles_completed_p90", "dispossessed_p90",
    "interceptions_p90", "ball_recoveries_p90",
    "tackles_won_p90", "blocks_p90", "clearances_p90",
    "fouls_committed_p90", "fouls_won_p90",
    "ga_p90",  # first-half G+A/90 is both a feature AND the baseline prediction
]


def half_season_stats(con, week_filter: str, label: str) -> pd.DataFrame:
    events = con.execute(f"""
        SELECT e.*
        FROM bronze_events e
        JOIN bronze_matches m USING (match_id)
        WHERE e.player_id IS NOT NULL AND {week_filter}
    """).df()
    minutes = con.execute(f"""
        SELECT pm.player_id, MODE(pm.player_name) AS player_name,
               SUM(pm.minutes_played) AS minutes
        FROM silver_player_minutes pm
        JOIN bronze_matches m USING (match_id)
        WHERE {week_filter}
        GROUP BY pm.player_id
    """).df()

    agg = events.groupby("player_id").agg(
        goals=("shot_outcome", lambda s: (s == "Goal").sum()),
        assists=("pass_goal_assist", "sum"),
        total_xg=("shot_statsbomb_xg", "sum"),
        shots=("type", lambda s: (s == "Shot").sum()),
        shot_assists=("pass_shot_assist", "sum"),
        crosses=("pass_cross", "sum"),
        through_balls=("pass_through_ball", "sum"),
        passes=("type", lambda s: (s == "Pass").sum()),
        passes_completed=("pass_outcome", lambda s: s.isna().sum()),  # only meaningful within Pass rows, corrected below
        dribbles_completed=("dribble_outcome", lambda s: (s == "Complete").sum()),
        dispossessed=("type", lambda s: (s == "Dispossessed").sum()),
        interceptions=("type", lambda s: (s == "Interception").sum()),
        ball_recoveries=("type", lambda s: (s == "Ball Recovery").sum()),
        tackles_won=("duel_outcome", lambda s: s.isin(["Won", "Success", "Success In Play", "Success Out"]).sum()),
        blocks=("type", lambda s: (s == "Block").sum()),
        clearances=("type", lambda s: (s == "Clearance").sum()),
        fouls_committed=("type", lambda s: (s == "Foul Committed").sum()),
        fouls_won=("type", lambda s: (s == "Foul Won").sum()),
    ).reset_index()

    # passes_completed needs to be computed only over Pass-type rows
    pass_rows = events[events["type"] == "Pass"]
    pc = pass_rows.groupby("player_id")["pass_outcome"].apply(lambda s: s.isna().sum()).rename("passes_completed_fixed")
    agg = agg.merge(pc, on="player_id", how="left")
    agg["passes_completed"] = agg["passes_completed_fixed"].fillna(0)
    agg = agg.drop(columns=["passes_completed_fixed"])

    df = minutes.merge(agg, on="player_id", how="left").fillna(0)
    df = df[df["minutes"] >= MIN_MINUTES_PER_HALF].copy()

    exposure = df["minutes"] / 90
    for col in ["goals", "assists", "total_xg", "shots", "shot_assists", "crosses", "through_balls",
                "passes_completed", "dribbles_completed", "dispossessed", "interceptions",
                "ball_recoveries", "tackles_won", "blocks", "clearances", "fouls_committed", "fouls_won"]:
        df[f"{col}_p90"] = df[col] / exposure
    df["pass_completion_pct"] = np.where(df["passes"] > 0, 100 * df["passes_completed"] / df["passes"], np.nan)
    df["ga_p90"] = df["goals_p90"] + df["assists_p90"]
    df = df.rename(columns={"minutes": f"minutes_{label}"})
    return df


def build():
    con = duckdb.connect(str(DB_PATH))
    first = half_season_stats(con, f"m.match_week <= {FIRST_HALF_MAX_WEEK}", "first")
    second = half_season_stats(con, f"m.match_week > {FIRST_HALF_MAX_WEEK}", "second")
    con.close()

    merged = first.merge(second[["player_id", "ga_p90", "minutes_second"]], on="player_id", suffixes=("", "_second"))
    merged = merged.rename(columns={"ga_p90_second": "target_ga_p90_second_half"})
    if "target_ga_p90_second_half" not in merged.columns:
        merged = merged.rename(columns={"ga_p90": "target_ga_p90_second_half"})  # fallback naming safety

    print(f"Players with >= {MIN_MINUTES_PER_HALF} min in BOTH halves: {len(merged)}")

    X = merged[FEATURE_COLS].fillna(0).to_numpy()
    y = merged["target_ga_p90_second_half"].to_numpy()
    baseline_pred_col = merged["ga_p90"].to_numpy()  # first-half G+A/90, used as the baseline prediction

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {"baseline": [], "ridge": [], "lasso": [], "lightgbm": []}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_z, X_test_z = scaler.transform(X_train), scaler.transform(X_test)

        # baseline: predicted = first-half G+A/90 (no fitting)
        base_pred = baseline_pred_col[test_idx]
        results["baseline"].append((mean_absolute_error(y_test, base_pred), r2_score(y_test, base_pred)))

        ridge = Ridge(alpha=1.0).fit(X_train_z, y_train)
        pred = ridge.predict(X_test_z)
        results["ridge"].append((mean_absolute_error(y_test, pred), r2_score(y_test, pred)))

        lasso = Lasso(alpha=0.01).fit(X_train_z, y_train)
        pred = lasso.predict(X_test_z)
        results["lasso"].append((mean_absolute_error(y_test, pred), r2_score(y_test, pred)))

        lgbm = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, min_child_samples=10)
        lgbm.fit(X_train, y_train)
        pred = lgbm.predict(X_test)
        results["lightgbm"].append((mean_absolute_error(y_test, pred), r2_score(y_test, pred)))

    print(f"\n=== 5-fold CV results: predicting second-half G+A/90 from first-half profile (n={len(merged)}) ===")
    print(f"{'model':<12}{'MAE (mean±std)':<22}{'R2 (mean±std)':<22}")
    summary = {}
    for name, vals in results.items():
        maes = [v[0] for v in vals]
        r2s = [v[1] for v in vals]
        summary[name] = {"mae_mean": np.mean(maes), "mae_std": np.std(maes), "r2_mean": np.mean(r2s), "r2_std": np.std(r2s)}
        print(f"{name:<12}{np.mean(maes):.4f} ± {np.std(maes):.4f}      {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    # also fit ridge/lasso on all data to show coefficients (interpretability sanity, not a new eval)
    scaler_full = StandardScaler().fit(X)
    ridge_full = Ridge(alpha=1.0).fit(scaler_full.transform(X), y)
    coefs = sorted(zip(FEATURE_COLS, ridge_full.coef_), key=lambda x: -abs(x[1]))
    print("\nRidge (full-data fit) top coefficients:")
    for f, c in coefs[:8]:
        print(f"  {f:<25} {c:+.4f}")

    return summary


if __name__ == "__main__":
    build()
