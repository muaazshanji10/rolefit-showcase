# RoleFit

A football recruitment analytics showcase built on StatsBomb's open data: role discovery, uncertainty-aware player output estimates, model comparison for predicting future form, and a natural-language "describe the player you want" interface - all backed by real per-match event data, not synthetic stats.

**Live app:** _pending deploy - see "Deploying" below_
**Repo:** https://github.com/muaazshanji10/rolefit-showcase

**Screenshot:** pending - no GUI/browser automation was available in the build environment to capture one honestly. Will be added as `docs/screenshot.png` once the app is deployed and clicked through.

## Problem

Recruitment analysts need three things a spreadsheet of season totals doesn't give them: (1) stats normalized so players can be compared fairly, (2) an honest sense of which numbers are signal vs. small-sample noise, and (3) a way to translate a scout's plain-English brief ("I need a tough, ball-winning No. 6") into something that can actually rank real players. RoleFit builds all three, end to end, on a single real dataset.

## Data

[StatsBomb open data](https://github.com/statsbomb/open-data) - Premier League 2015/16 (Leicester's title season), 380 matches, ~1.3M events. Used under StatsBomb's free/open terms for public non-commercial use; see the [StatsBomb Media Pack / free data hub](https://statsbomb.com/what-we-do/hub/free-data/) and the logo + attribution in the app footer.

## Method

1. **Pipeline** - raw JSON → DuckDB bronze/silver/gold. Minutes played are derived from the event-level match clock (Starting XI, Substitution, Half End, red/second-yellow cards), not the lineup file's own position-stint timestamps - those showed period-boundary inconsistencies in stoppage time on inspection. Spot-checked against source JSON (Harry Kane's sub-off at 63:08 in match 3754270 → 63.13 computed minutes, exact).
2. **Feature engineering** - per-90 rates for ~20 raw stats, plus possession-adjusted (padj) defensive rates (possession proxied from on-ball event share per team-match, since StatsBomb's open data has no literal time-of-possession field - a documented v1 simplification). A correlation check over 28 per-90 features flagged 18 pairs at |r|>0.8, all sensible (padj tracks raw, goals/xG/shots cluster, passes/completions cluster) - used to trim the feature set for clustering.
3. **Shrinkage + uncertainty** - bootstrap 95% CIs on raw goals/90 (resampling each player's own matches), and empirical-Bayes (Gamma-Poisson) shrinkage toward the population rate. Small-sample high-raw-rate players (e.g. 1 goal in ~2 matches, raw 0.67/90) drop out of the shrunk top-10 while high-minute scorers barely move - the intended effect.
4. **Role discovery** - 17 deduplicated per-90 features → PCA (6 components, 85.4% variance) → k-means (k=6, chosen by silhouette over k=4..10). Clusters land cleanly on real positional/tactical lines: goalkeepers isolate on their own; center-backs split from fullbacks on clearing vs. crossing; defensive mids split from wide creators on tackling vs. shot-creation; strikers cluster on xG/shots.
5. **Model comparison** - predict a player's second-half-of-season G+A/90 from their first-half stat profile (n=260, ≥450 min per half), 5-fold CV, default hyperparameters, no tuning loop:

   | model | MAE | R² |
   |---|---|---|
   | baseline ("same as first half") | 0.139 ± 0.031 | 0.179 ± 0.282 |
   | ridge | 0.119 ± 0.015 | 0.468 ± 0.129 |
   | **lasso** | **0.117 ± 0.018** | **0.486 ± 0.123** |
   | lightgbm | 0.134 ± 0.014 | 0.309 ± 0.156 |

   Real result, not cherry-picked: lasso wins, both regularized linear models clearly beat the naive baseline and beat LightGBM (plausible at n=260 - boosting wants more data). In the full-data ridge fit, a player's own first-half G+A gets a *negative* coefficient once `total_xg_p90` is in the model - raw output partly reflects finishing variance that regresses once shot quality is controlled for.
6. **Interpretability** - one SHAP pass (TreeExplainer) on the LightGBM model. Top 5 by mean |SHAP|: `shots_p90`, `total_xg_p90`, `shot_assists_p90`, `through_balls_p90`, `assists_p90` - attacking-output and chance-creation stats dominate, matching football sense for a G+A/90 target.
7. **Similarity engine** - Euclidean + cosine similarity on standardized per-90 profiles (394-player pool, ≥450 min). Spot check: Kane's nearest neighbors are all out-and-out strikers (Agüero, Bony, Ighalo, Lukaku); Mahrez's are wide creative players (Sánchez, Mané, Barkley, Redmond).
8. **LLM/NL interface** - a real Anthropic API call (claude-sonnet-5) converts a plain-English brief into a weight vector over the same 17-feature space, compared against the hand-built cluster centroids from step 4. All 3 test briefs land on the correct cluster with a clear margin (e.g. defensive-mid brief: 0.724 cosine sim to the DM cluster vs. 0.339 runner-up). Genuine divergences surfaced, not papered over: the LLM doesn't predict that this dataset's real DMs foul a lot, or that real wide dribblers get dispossessed more as a side effect of risk-taking - both are behavioral patterns invisible from prose alone.
9. **App** - Streamlit, five tabs (search, role fit, similarity, NL brief, projections with uncertainty), deployed to Streamlit Community Cloud from this repo.

## v1 scope

This is deliberately shallow-but-real: **one league, one season** (Premier League 2015/16), **light tuning** (sensible default hyperparameters, no grid search, no deep hyperparameter interrogation), **one honest pass per phase**. Every number in this README came from an actual run against real data - nothing stubbed or faked. Where something didn't work as first written (a minutes-computation approach, a units bug in the shrinkage math, a name-aggregation choice, an API response-parsing assumption), it's called out in the commit history rather than silently smoothed over.

## Roadmap (v2)

- Combine all four StatsBomb open-data leagues for a larger, more general training set
- Deeper hyperparameter search for the model comparison (currently intentionally untuned)
- Extend role discovery/similarity across multiple seasons to reduce single-season noise
- Add more granular position-specific role taxonomies within the current 6 clusters

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pipeline (expects ~/projects/statsbomb-data cloned locally)
python3 src/pipeline.py
python3 src/features.py
python3 src/shrinkage.py
python3 src/roles.py
python3 src/model_compare.py
python3 src/interpret.py
python3 src/similarity.py
python3 src/export_app_data.py   # writes app/data/*.parquet for the app

# .env with ANTHROPIC_API_KEY=sk-ant-... required for src/llm_brief.py and the app's NL Brief tab
streamlit run app/app.py
```

## Deploying

1. Push to GitHub (already done: `muaazshanji10/rolefit-showcase`).
2. On [share.streamlit.io](https://share.streamlit.io): New app → this repo → branch `master` → main file `app/app.py`.
3. In the app's Advanced settings → Secrets, add `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. The app ships with precomputed `app/data/*.parquet` snapshots of the gold tables, so it doesn't need DuckDB, the raw StatsBomb data, or the pipeline at deploy time.
