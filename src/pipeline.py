"""
RoleFit pipeline: StatsBomb open-data JSON -> DuckDB bronze/silver/gold.

Scope (v1): Premier League 2015/16 only (competition_id=2, season_id=27).

Minutes-played method: derived from events (Starting XI, Substitution,
Half End, and card events), NOT from lineups[].positions[].from/to —
those raw timestamps show period-boundary inconsistencies (stoppage-time
entries mislabelled with the next period's index) that make them
unreliable for a v1 pass. Event `minute`/`second` fields are the
cumulative, monotonic match clock StatsBomb itself uses for stoppage
time, so they are the trustworthy source here.

Date parsing gotcha: match_date in matches/*.json is already clean
ISO (YYYY-MM-DD). We parse it with an explicit format, never
format="mixed", since "mixed" can silently misinterpret ambiguous
date strings.
"""
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

STATSBOMB_ROOT = Path(os.path.expanduser("~/projects/statsbomb-data/data"))
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rolefit.duckdb"
COMPETITION_ID = 2
SEASON_ID = 27  # Premier League 2015/16


def load_matches() -> pd.DataFrame:
    matches = json.load(open(STATSBOMB_ROOT / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json"))
    rows = []
    for m in matches:
        rows.append({
            "match_id": m["match_id"],
            "match_date": m["match_date"],
            "kick_off": m.get("kick_off"),
            "home_team_id": m["home_team"]["home_team_id"],
            "home_team_name": m["home_team"]["home_team_name"],
            "away_team_id": m["away_team"]["away_team_id"],
            "away_team_name": m["away_team"]["away_team_name"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "match_week": m.get("match_week"),
            "competition_stage": m.get("competition_stage", {}).get("name"),
            "stadium": (m.get("stadium") or {}).get("name"),
            "referee": (m.get("referee") or {}).get("name"),
        })
    df = pd.DataFrame(rows)
    # Explicit format, never "mixed" -- match_date is clean ISO (YYYY-MM-DD).
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y-%m-%d")
    return df


def load_lineups(match_ids: list[int]) -> pd.DataFrame:
    rows = []
    for mid in match_ids:
        teams = json.load(open(STATSBOMB_ROOT / "lineups" / f"{mid}.json"))
        for team in teams:
            for p in team["lineup"]:
                primary_pos = p["positions"][0]["position"] if p["positions"] else None
                rows.append({
                    "match_id": mid,
                    "team_id": team["team_id"],
                    "team_name": team["team_name"],
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "player_nickname": p["player_nickname"],
                    "jersey_number": p["jersey_number"],
                    "country": (p.get("country") or {}).get("name"),
                    "primary_position": primary_pos,
                    "started": bool(p["positions"]) and p["positions"][0]["start_reason"] == "Starting XI",
                    "n_cards": len(p.get("cards", [])),
                })
    return pd.DataFrame(rows)


def load_events(match_ids: list[int]) -> pd.DataFrame:
    rows = []
    for mid in match_ids:
        events = json.load(open(STATSBOMB_ROOT / "events" / f"{mid}.json"))
        for e in events:
            player = e.get("player") or {}
            team = e.get("team") or {}
            position = e.get("position") or {}
            type_ = e.get("type") or {}
            loc = e.get("location")
            shot = e.get("shot") or {}
            pass_ = e.get("pass") or {}
            duel = e.get("duel") or {}
            foul = e.get("foul_committed") or {}
            bad_behaviour = e.get("bad_behaviour") or {}
            carry = e.get("carry") or {}
            dribble = e.get("dribble") or {}
            rows.append({
                "match_id": mid,
                "event_id": e["id"],
                "index": e["index"],
                "period": e["period"],
                "minute": e["minute"],
                "second": e["second"],
                "type": type_.get("name"),
                "possession_team": (e.get("possession_team") or {}).get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "position_name": position.get("name"),
                "location_x": loc[0] if loc else None,
                "location_y": loc[1] if loc else None,
                "duration": e.get("duration"),
                "under_pressure": e.get("under_pressure", False),
                # shot
                "shot_statsbomb_xg": shot.get("statsbomb_xg"),
                "shot_outcome": (shot.get("outcome") or {}).get("name"),
                "shot_type": (shot.get("type") or {}).get("name"),
                "shot_body_part": (shot.get("body_part") or {}).get("name"),
                # pass
                "pass_outcome": (pass_.get("outcome") or {}).get("name"),  # None == complete
                "pass_length": pass_.get("length"),
                "pass_angle": pass_.get("angle"),
                "pass_height": (pass_.get("height") or {}).get("name"),
                "pass_goal_assist": pass_.get("goal_assist", False),
                "pass_shot_assist": pass_.get("shot_assist", False),
                "pass_cross": pass_.get("cross", False),
                "pass_through_ball": pass_.get("through_ball", False),
                "pass_switch": pass_.get("switch", False),
                "pass_recipient_id": (pass_.get("recipient") or {}).get("id"),
                # duel / defensive
                "duel_type": (duel.get("type") or {}).get("name"),
                "duel_outcome": (duel.get("outcome") or {}).get("name"),
                # carry / dribble
                "carry_end_x": (carry.get("end_location") or [None, None])[0],
                "carry_end_y": (carry.get("end_location") or [None, None])[1],
                "dribble_outcome": (dribble.get("outcome") or {}).get("name"),
                # discipline
                "foul_card": (foul.get("card") or {}).get("name"),
                "bad_behaviour_card": (bad_behaviour.get("card") or {}).get("name"),
                # substitution
                "sub_replacement_id": (e.get("substitution") or {}).get("replacement", {}).get("id"),
                "sub_replacement_name": (e.get("substitution") or {}).get("replacement", {}).get("name"),
            })
    return pd.DataFrame(rows)


def compute_minutes(events: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Per (match_id, player_id): minutes played, derived from event clock."""
    rows = []
    for mid, ev in events.groupby("match_id"):
        half_ends = ev[ev["type"] == "Half End"]
        match_end = (half_ends["minute"] + half_ends["second"] / 60).max()
        if pd.isna(match_end):
            match_end = (ev["minute"] + ev["second"] / 60).max()

        starters = lineups[(lineups["match_id"] == mid) & (lineups["started"])]
        start_time = {pid: 0.0 for pid in starters["player_id"]}

        subs = ev[ev["type"] == "Substitution"].copy()
        subs_off_time = {}
        for _, s in subs.iterrows():
            t = s["minute"] + s["second"] / 60
            subs_off_time[s["player_id"]] = t
            if pd.notna(s["sub_replacement_id"]):
                start_time[int(s["sub_replacement_id"])] = t

        # red cards / second yellow end a player's involvement early
        red_time = {}
        cards = ev[ev["foul_card"].isin(["Red Card", "Second Yellow"]) | (ev["bad_behaviour_card"] == "Red Card")]
        for _, c in cards.iterrows():
            if pd.notna(c["player_id"]):
                t = c["minute"] + c["second"] / 60
                red_time[c["player_id"]] = min(red_time.get(c["player_id"], t), t)

        all_players = set(start_time) | set(subs_off_time)
        for pid in all_players:
            start = start_time.get(pid, 0.0)
            end_candidates = [match_end]
            if pid in subs_off_time:
                end_candidates.append(subs_off_time[pid])
            if pid in red_time:
                end_candidates.append(red_time[pid])
            end = min(end_candidates)
            rows.append({"match_id": mid, "player_id": pid, "minutes_played": max(end - start, 0.0)})
    return pd.DataFrame(rows)


def build():
    print("Loading matches...")
    matches = load_matches()
    match_ids = matches["match_id"].tolist()
    print(f"{len(match_ids)} matches")

    print("Loading lineups...")
    lineups = load_lineups(match_ids)
    print(f"{len(lineups)} lineup rows")

    print("Loading events (this takes a minute for 380 matches)...")
    events = load_events(match_ids)
    print(f"{len(events)} event rows")

    print("Computing minutes played...")
    minutes = compute_minutes(events, lineups)
    print(f"{len(minutes)} player-match minutes rows")

    DB_PATH.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # bronze: raw-ish flattened tables
    con.execute("CREATE OR REPLACE TABLE bronze_matches AS SELECT * FROM matches")
    con.execute("CREATE OR REPLACE TABLE bronze_lineups AS SELECT * FROM lineups")
    con.execute("CREATE OR REPLACE TABLE bronze_events AS SELECT * FROM events")

    # silver: cleaned, joined, typed
    con.execute("""
        CREATE OR REPLACE TABLE silver_player_minutes AS
        SELECT
            m.player_id,
            m.match_id,
            mt.match_date,
            mt.home_team_name,
            mt.away_team_name,
            l.team_name,
            l.player_name,
            l.primary_position,
            m.minutes_played
        FROM minutes m
        JOIN bronze_matches mt USING (match_id)
        JOIN bronze_lineups l ON l.match_id = m.match_id AND l.player_id = m.player_id
    """)

    con.execute("""
        CREATE OR REPLACE TABLE silver_events AS
        SELECT e.*, mt.match_date
        FROM bronze_events e
        JOIN bronze_matches mt USING (match_id)
        WHERE e.player_id IS NOT NULL
    """)

    # gold: season-level per-player aggregates (raw counts; per-90 done in feature step)
    con.execute("""
        CREATE OR REPLACE TABLE gold_player_season AS
        WITH minutes_agg AS (
            SELECT player_id, ANY_VALUE(player_name) AS player_name,
                   COUNT(DISTINCT match_id) AS matches_played,
                   SUM(minutes_played) AS total_minutes,
                   MODE(primary_position) AS primary_position
            FROM silver_player_minutes
            GROUP BY player_id
        ),
        event_agg AS (
            SELECT
                player_id,
                COUNT(*) FILTER (WHERE type = 'Pass') AS passes,
                COUNT(*) FILTER (WHERE type = 'Pass' AND pass_outcome IS NULL) AS passes_completed,
                COUNT(*) FILTER (WHERE pass_goal_assist) AS assists,
                COUNT(*) FILTER (WHERE pass_shot_assist) AS shot_assists,
                COUNT(*) FILTER (WHERE pass_cross) AS crosses,
                COUNT(*) FILTER (WHERE pass_through_ball) AS through_balls,
                COUNT(*) FILTER (WHERE type = 'Shot') AS shots,
                COUNT(*) FILTER (WHERE type = 'Shot' AND shot_outcome = 'Goal') AS goals,
                SUM(shot_statsbomb_xg) FILTER (WHERE type = 'Shot') AS total_xg,
                COUNT(*) FILTER (WHERE type = 'Dribble') AS dribbles,
                COUNT(*) FILTER (WHERE type = 'Dribble' AND dribble_outcome = 'Complete') AS dribbles_completed,
                COUNT(*) FILTER (WHERE type = 'Interception') AS interceptions,
                COUNT(*) FILTER (WHERE type = 'Ball Recovery') AS ball_recoveries,
                COUNT(*) FILTER (WHERE type = 'Duel' AND duel_type = 'Tackle') AS tackles,
                COUNT(*) FILTER (WHERE type = 'Duel' AND duel_type = 'Tackle' AND duel_outcome IN ('Won','Success','Success In Play','Success Out')) AS tackles_won,
                COUNT(*) FILTER (WHERE type = 'Block') AS blocks,
                COUNT(*) FILTER (WHERE type = 'Clearance') AS clearances,
                COUNT(*) FILTER (WHERE type = 'Foul Committed') AS fouls_committed,
                COUNT(*) FILTER (WHERE type = 'Foul Won') AS fouls_won,
                COUNT(*) FILTER (WHERE type = 'Miscontrol') AS miscontrols,
                COUNT(*) FILTER (WHERE type = 'Dispossessed') AS dispossessed
            FROM silver_events
            GROUP BY player_id
        )
        SELECT ma.*, ea.* EXCLUDE (player_id)
        FROM minutes_agg ma
        LEFT JOIN event_agg ea USING (player_id)
        ORDER BY total_minutes DESC
    """)

    con.close()
    print(f"DuckDB written to {DB_PATH}")


if __name__ == "__main__":
    build()
