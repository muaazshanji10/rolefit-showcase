"""
RoleFit similarity engine (Phase 7): "find players like X" via
Euclidean distance and cosine similarity on standardized per-90 profiles.

Pool: players with >=450 minutes (same threshold as Phase 5's model
data), using the same 17-feature set as Phase 4's role discovery so
role clusters and similarity search are built on the same footing.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from roles import FEATURES

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
MIN_MINUTES = 450


def load_pool():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold_player_features").df()
    con.close()
    df = df[df["total_minutes"] >= MIN_MINUTES].reset_index(drop=True)
    X = df[FEATURES].fillna(0)
    Xz = StandardScaler().fit_transform(X)
    return df, Xz


def find_similar(player_name: str, df: pd.DataFrame, Xz: np.ndarray, n=5):
    matches = df.index[df["player_name"] == player_name].tolist()
    if not matches:
        candidates = df[df["player_name"].str.contains(player_name, case=False, na=False)]["player_name"].tolist()
        raise ValueError(f"'{player_name}' not found in pool (>= {MIN_MINUTES} min). Did you mean: {candidates[:5]}?")
    idx = matches[0]

    euc = euclidean_distances(Xz[[idx]], Xz)[0]
    cos = cosine_similarity(Xz[[idx]], Xz)[0]

    euc_order = np.argsort(euc)
    euc_order = euc_order[euc_order != idx][:n]
    cos_order = np.argsort(-cos)
    cos_order = cos_order[cos_order != idx][:n]

    euc_df = df.iloc[euc_order][["player_name", "primary_position", "total_minutes"]].copy()
    euc_df["euclidean_dist"] = euc[euc_order]

    cos_df = df.iloc[cos_order][["player_name", "primary_position", "total_minutes"]].copy()
    cos_df["cosine_sim"] = cos[cos_order]

    return euc_df, cos_df


if __name__ == "__main__":
    df, Xz = load_pool()
    print(f"Similarity pool: {len(df)} players (>= {MIN_MINUTES} min)\n")

    for name in ["Harry Kane", "N'Golo Kanté", "Riyad Mahrez"]:
        try:
            euc_df, cos_df = find_similar(name, df, Xz)
        except ValueError as e:
            print(e)
            continue
        print(f"=== Players like {name} (Euclidean) ===")
        print(euc_df.to_string(index=False))
        print(f"\n=== Players like {name} (Cosine) ===")
        print(cos_df.to_string(index=False))
        print()
