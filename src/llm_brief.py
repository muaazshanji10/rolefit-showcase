"""
RoleFit LLM/NL interface (Phase 8).

Converts a plain-English scouting brief into a structured weight vector
over the same 17-feature space used for role discovery (Phase 4) and
similarity (Phase 7), via a single real Anthropic API call. Then
compares the LLM's weights against the hand-built cluster centroids
from Phase 4, reporting where they agree and where they diverge --
not skipped, not faked.
"""
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from anthropic import Anthropic

from roles import FEATURES, MIN_MINUTES

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
MODEL = "claude-sonnet-5"

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
    "interceptions_p90": "interceptions per 90 minutes, possession-adjusted",
    "ball_recoveries_p90": "loose-ball recoveries per 90 minutes, possession-adjusted",
    "tackles_won_p90": "tackles won per 90 minutes, possession-adjusted",
    "blocks_p90": "shots/passes blocked per 90 minutes, possession-adjusted",
    "clearances_p90": "defensive clearances per 90 minutes, possession-adjusted",
    "fouls_committed_p90": "fouls committed per 90 minutes",
    "fouls_won_p90": "fouls won per 90 minutes",
}
# note: the 5 possession-adjusted features are actually named *_padj_p90 in the
# data; described above under their plain name for LLM readability, mapped back below.
PADJ_MAP = {
    "interceptions_p90": "interceptions_padj_p90",
    "ball_recoveries_p90": "ball_recoveries_padj_p90",
    "tackles_won_p90": "tackles_won_padj_p90",
    "blocks_p90": "blocks_padj_p90",
    "clearances_p90": "clearances_padj_p90",
}

SYSTEM_PROMPT = f"""You are a football (soccer) recruitment analyst. You convert a plain-English \
scouting brief into a structured weight vector over a fixed set of per-90-minute statistical \
features, so the weights can be used to numerically score and rank real players.

Available features (name: meaning):
{chr(10).join(f"- {k}: {v}" for k, v in FEATURE_DESCRIPTIONS.items())}

Return ONLY a JSON object mapping every one of the above feature names to a weight in [-1, 1]:
- Positive weight: this stat matters and more of it is better for the brief.
- Negative weight: this stat matters and LESS of it is better for the brief (e.g. dispossessed_p90 \
for a brief valuing ball security).
- Near zero: this stat is not relevant to the brief.
Weights should reflect RELATIVE importance -- the most defining traits of the brief should have the \
largest magnitude. Output strictly valid JSON, no prose, no markdown fences."""


def brief_to_weights(brief: str) -> dict:
    client = Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Brief: {brief}"}],
    )
    text = next(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    weights = json.loads(text)
    # map plain names back to actual padj column names
    mapped = {}
    for k, v in weights.items():
        col = PADJ_MAP.get(k, k)
        mapped[col] = float(v)
    return mapped


def load_cluster_centroids():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM gold_player_roles").df()
    con.close()
    X = df[FEATURES].fillna(0)
    Xz = StandardScaler().fit_transform(X)
    Xz_df = pd.DataFrame(Xz, columns=FEATURES, index=df.index)
    Xz_df["cluster"] = df["cluster"].values
    Xz_df["cluster_name"] = df["cluster_name"].values
    centroids = Xz_df.groupby(["cluster", "cluster_name"])[FEATURES].mean()
    return centroids


def compare_to_clusters(weights: dict, centroids: pd.DataFrame):
    w = np.array([weights.get(f, 0.0) for f in FEATURES])
    w_norm = w / (np.linalg.norm(w) + 1e-9)
    sims = {}
    for (cid, cname), row in centroids.iterrows():
        c = row.to_numpy()
        c_norm = c / (np.linalg.norm(c) + 1e-9)
        sims[(cid, cname)] = float(np.dot(w_norm, c_norm))
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    return ranked


def run_brief(brief: str, centroids: pd.DataFrame):
    print(f"\n{'='*70}\nBRIEF: {brief}\n{'='*70}")
    weights = brief_to_weights(brief)

    top_llm = sorted(weights.items(), key=lambda x: -abs(x[1]))[:6]
    print("LLM top weighted features:")
    for f, v in top_llm:
        print(f"  {f:<28} {v:+.2f}")

    ranked = compare_to_clusters(weights, centroids)
    print("\nClosest hand-built cluster(s) by cosine similarity to LLM weight vector:")
    for (cid, cname), sim in ranked[:3]:
        print(f"  cluster {cid} ({cname}): sim={sim:.3f}")

    best_cid, best_cname = ranked[0][0]
    best_centroid = centroids.loc[(best_cid, best_cname)]
    top_cluster_feats = best_centroid.sort_values(ascending=False).head(5)
    print(f"\nBest-match cluster {best_cid} top mean z-scored features (hand-built):")
    for f, z in top_cluster_feats.items():
        print(f"  {f:<28} z={z:+.2f}")

    llm_top_set = {f for f, _ in top_llm if _ != 0}
    cluster_top_set = set(top_cluster_feats.index)
    agree = llm_top_set & cluster_top_set
    diverge_llm_only = llm_top_set - cluster_top_set
    diverge_cluster_only = cluster_top_set - llm_top_set
    print(f"\nAGREE on: {sorted(agree) if agree else '(none)'}")
    print(f"LLM emphasizes but cluster doesn't (top-5): {sorted(diverge_llm_only) if diverge_llm_only else '(none)'}")
    print(f"Cluster emphasizes but LLM top-6 doesn't:   {sorted(diverge_cluster_only) if diverge_cluster_only else '(none)'}")

    return weights, ranked


if __name__ == "__main__":
    centroids = load_cluster_centroids()
    print("Hand-built cluster names (Phase 4):")
    for (cid, cname) in centroids.index:
        print(f"  {cid}: {cname}")

    briefs = [
        "I need a tough, ball-winning defensive midfielder who breaks up opposition attacks and keeps possession simple and secure.",
        "Looking for a clinical goal-scoring striker who gets on the end of chances and shoots frequently.",
        "Want a creative wide attacker who can beat a full-back 1v1 and create chances with through balls and crosses.",
    ]
    for b in briefs:
        run_brief(b, centroids)
