"""
RoleFit shrinkage + uncertainty (Phase 3).

Demonstrated on goals per 90 -- the stat where small-sample noise is most
visible to a football audience (a sub with 1 goal in 20 minutes has a raw
rate of 4.5 goals/90, which nobody would take at face value).

Two independent pieces, both v1-honest simplifications:

1. Bootstrap CI on the RAW per-90 rate: resample a player's own matches
   with replacement (1000 draws), recompute the rate each time, take the
   2.5/97.5 percentiles. Players with very few matches get bootstrap CIs
   that are degenerate or absurdly wide -- that's not a bug, it's the
   point: it's the mechanism showing why the number can't be trusted yet.

2. Empirical-Bayes shrinkage (Gamma-Poisson conjugate): treat each
   player's goal count as Poisson(rate * exposure), rate ~ Gamma(a, b)
   with (a, b) estimated from the population via method-of-moments.
   Posterior mean rate = (a + goals) / (b + exposure) -- pulls
   small-exposure players toward the population mean, low-exposure high
   raw-rate players shrink hardest.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
N_BOOT = 1000
RNG = np.random.default_rng(42)


def load_player_match_goals(con) -> pd.DataFrame:
    goals = con.execute("""
        SELECT match_id, player_id, COUNT(*) AS goals
        FROM bronze_events
        WHERE type = 'Shot' AND shot_outcome = 'Goal'
        GROUP BY match_id, player_id
    """).df()
    minutes = con.execute("SELECT match_id, player_id, player_name, minutes_played FROM silver_player_minutes").df()
    merged = minutes.merge(goals, on=["match_id", "player_id"], how="left")
    merged["goals"] = merged["goals"].fillna(0).astype(int)
    return merged


def bootstrap_ci(group: pd.DataFrame, n_boot=N_BOOT):
    mins = group["minutes_played"].to_numpy()
    gls = group["goals"].to_numpy()
    n = len(mins)
    if mins.sum() == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_mins = mins[idx].sum(axis=1)
    boot_goals = gls[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(boot_mins > 0, boot_goals * 90 / boot_mins, np.nan)
    rates = rates[~np.isnan(rates)]
    if len(rates) == 0:
        return np.nan, np.nan
    return np.percentile(rates, 2.5), np.percentile(rates, 97.5)


def empirical_bayes_shrink(season: pd.DataFrame) -> pd.DataFrame:
    season = season.copy()
    season["exposure"] = season["total_minutes"] / 90
    season = season[season["exposure"] > 0]

    mu = season["goals"].sum() / season["exposure"].sum()
    raw_rate = season["goals"] / season["exposure"]
    weighted_var = np.average((raw_rate - mu) ** 2, weights=season["exposure"])
    sampling_var_avg = mu * np.average(1 / season["exposure"], weights=season["exposure"])
    prior_var = max(weighted_var - sampling_var_avg, mu**2 / 100)  # floor to keep alpha/beta well-defined

    alpha = mu**2 / prior_var
    beta = mu / prior_var

    # mu, raw_rate, and the posterior mean below are already in per-90 units
    # (exposure is minutes/90) -- do not multiply by 90 again.
    season["raw_p90"] = raw_rate
    season["shrunk_p90"] = (alpha + season["goals"]) / (beta + season["exposure"])
    print(f"Population mean goals/90: {mu:.4f}")
    print(f"Empirical-Bayes prior: Gamma(alpha={alpha:.2f}, beta={beta:.2f}) [rate is per-90-equivalent]")
    return season


def build():
    con = duckdb.connect(str(DB_PATH))
    pm_goals = load_player_match_goals(con)
    season = con.execute("""
        SELECT player_id, player_name, primary_position, matches_played, total_minutes, goals
        FROM gold_player_season
    """).df()
    season["goals"] = season["goals"].fillna(0).astype(float)

    print("Computing per-player bootstrap CIs (this takes a bit for 550 players)...")
    ci_rows = []
    for pid, g in pm_goals.groupby("player_id"):
        lo, hi = bootstrap_ci(g)
        ci_rows.append({"player_id": pid, "boot_ci_low": lo, "boot_ci_high": hi})
    ci_df = pd.DataFrame(ci_rows)

    shrunk = empirical_bayes_shrink(season)
    result = shrunk.merge(ci_df, on="player_id", how="left")

    con.execute("CREATE OR REPLACE TABLE gold_goals_shrinkage AS SELECT * FROM result")
    con.close()

    min_minutes = 90
    eligible = result[result["total_minutes"] >= min_minutes]

    print(f"\n=== TOP 10 by RAW goals/90 (min {min_minutes} min played) ===")
    top_raw = eligible.sort_values("raw_p90", ascending=False).head(10)
    print(top_raw[["player_name", "matches_played", "total_minutes", "goals", "raw_p90", "boot_ci_low", "boot_ci_high", "shrunk_p90"]].to_string(index=False))

    print(f"\n=== TOP 10 by SHRUNK goals/90 (min {min_minutes} min played) ===")
    top_shrunk = eligible.sort_values("shrunk_p90", ascending=False).head(10)
    print(top_shrunk[["player_name", "matches_played", "total_minutes", "goals", "raw_p90", "boot_ci_low", "boot_ci_high", "shrunk_p90"]].to_string(index=False))

    return result


if __name__ == "__main__":
    build()
