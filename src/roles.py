"""
RoleFit role discovery (Phase 4): PCA + k-means on per-90 features,
then name each cluster from its members and top loadings.

Feature set is trimmed based on the Phase 2 correlation check: keep one
side of each highly-correlated pair (padj variant over raw where both
exist, xG over goals as the more stable attacking signal, completed
counts over attempted where a %-complete companion exists).

Only players with >=900 minutes (~10 full matches) are clustered --
below that, per-90 rates are too noisy for role discovery to mean
anything (this is exactly the small-sample problem Phase 3 quantified).
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
MIN_MINUTES = 900

FEATURES = [
    "total_xg_p90", "shots_p90", "assists_p90", "shot_assists_p90",
    "crosses_p90", "through_balls_p90",
    "passes_completed_p90", "pass_completion_pct",
    "dribbles_completed_p90", "dispossessed_p90",
    "interceptions_padj_p90", "ball_recoveries_padj_p90",
    "tackles_won_padj_p90", "blocks_padj_p90", "clearances_padj_p90",
    "fouls_committed_p90", "fouls_won_p90",
]

K_RANGE = range(4, 11)


def pick_k(X_pca):
    scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_pca)
        scores[k] = silhouette_score(X_pca, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def name_cluster(df_cluster: pd.DataFrame, feature_z: pd.DataFrame, cluster_id: int) -> str:
    top_pos = df_cluster["primary_position"].value_counts().head(2)
    top_feats = feature_z.loc[df_cluster.index].mean().sort_values(ascending=False).head(3)
    pos_str = "/".join(top_pos.index[:2])
    feat_str = ", ".join(f"{f.replace('_p90','').replace('_padj','(padj)')}" for f in top_feats.index)
    return f"[{pos_str}] high: {feat_str}"


def build():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold_player_features").df()
    df = df[df["total_minutes"] >= MIN_MINUTES].reset_index(drop=True)
    print(f"Clustering {len(df)} players with >= {MIN_MINUTES} minutes")

    X = df[FEATURES].fillna(0)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    Xz_df = pd.DataFrame(Xz, columns=FEATURES, index=df.index)

    pca = PCA(n_components=0.85, random_state=42)
    X_pca = pca.fit_transform(Xz)
    print(f"PCA: {pca.n_components_} components explain {pca.explained_variance_ratio_.sum():.1%} variance")

    best_k, scores = pick_k(X_pca)
    print(f"Silhouette scores by k: {{{', '.join(f'{k}: {v:.3f}' for k, v in scores.items())}}}")
    print(f"Chosen k={best_k} (highest silhouette)")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X_pca)

    cluster_names = {}
    print("\n=== Clusters ===")
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        name = name_cluster(sub, Xz_df, cid)
        cluster_names[cid] = name
        top_players = sub.sort_values("total_minutes", ascending=False)["player_name"].head(5).tolist()
        print(f"\nCluster {cid} (n={len(sub)}): {name}")
        print(f"  Sample players: {', '.join(top_players)}")

    df["cluster_name"] = df["cluster"].map(cluster_names)

    con.execute("CREATE OR REPLACE TABLE gold_player_roles AS SELECT * FROM df")
    con.close()
    return df


if __name__ == "__main__":
    build()
