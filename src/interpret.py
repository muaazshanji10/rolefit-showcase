"""
RoleFit interpretability (Phase 6): one SHAP pass over the LightGBM
model from Phase 5, fit on the full dataset. Top features + a gut-check
against football sense. Not a deep interrogation -- one honest look.
"""
from pathlib import Path

import numpy as np
import shap
from lightgbm import LGBMRegressor

from model_compare import half_season_stats, FEATURE_COLS, FIRST_HALF_MAX_WEEK, MIN_MINUTES_PER_HALF
import duckdb

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"


def build():
    con = duckdb.connect(str(DB_PATH))
    first = half_season_stats(con, f"m.match_week <= {FIRST_HALF_MAX_WEEK}", "first")
    second = half_season_stats(con, f"m.match_week > {FIRST_HALF_MAX_WEEK}", "second")
    con.close()

    merged = first.merge(second[["player_id", "ga_p90"]], on="player_id", suffixes=("", "_second"))
    X = merged[FEATURE_COLS].fillna(0)
    y = merged["ga_p90_second"]

    model = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, min_child_samples=10)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLS, mean_abs_shap), key=lambda x: -x[1])

    print("=== SHAP: mean |impact| on predicted second-half G+A/90 ===")
    for f, v in ranked:
        print(f"  {f:<25} {v:.4f}")

    print("\nTop 5:", ", ".join(f[0] for f in ranked[:5]))
    return ranked


if __name__ == "__main__":
    build()
