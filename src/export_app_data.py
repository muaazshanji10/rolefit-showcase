"""
Export the small, derived gold tables to parquet under app/data/ so the
Streamlit app can run standalone on Streamlit Community Cloud without
DuckDB, the raw StatsBomb JSON, or re-running the pipeline. Only
aggregated/analytical outputs are exported -- raw and bronze/silver
data stays out of git per .gitignore.
"""
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
APP_DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"

TABLES = ["gold_player_features", "gold_player_roles", "gold_goals_shrinkage"]


def export():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    for t in TABLES:
        out = APP_DATA_DIR / f"{t}.parquet"
        con.execute(f"COPY {t} TO '{out}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n} rows -> {out} ({out.stat().st_size / 1024:.1f} KB)")
    con.close()


if __name__ == "__main__":
    export()
