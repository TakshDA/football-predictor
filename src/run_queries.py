"""
SQL Analysis Queries
Runs analysis queries against the project SQLite database.
Demonstrates joins, CTEs, window functions and conditional aggregation.

Requires data/football.db (create it with: py src/build_database.py)
"""

import os
import sqlite3
import pandas as pd

DB_FILE = os.path.join("data", "football.db")

QUERIES = {

"Home advantage by season": """
    SELECT Season,
           COUNT(*) AS matches,
           ROUND(100.0 * SUM(CASE WHEN FTR = 'H' THEN 1 ELSE 0 END) / COUNT(*), 1) AS home_win_pct,
           ROUND(100.0 * SUM(CASE WHEN FTR = 'D' THEN 1 ELSE 0 END) / COUNT(*), 1) AS draw_pct,
           ROUND(100.0 * SUM(CASE WHEN FTR = 'A' THEN 1 ELSE 0 END) / COUNT(*), 1) AS away_win_pct
    FROM matches
    WHERE FTR IS NOT NULL
    GROUP BY Season
    ORDER BY Season DESC
    LIMIT 6
""",

"All-time table (11 seasons)": """
    WITH team_matches AS (
        SELECT HomeTeam AS team, FTHG AS gf, FTAG AS ga,
               CASE FTR WHEN 'H' THEN 3 WHEN 'D' THEN 1 ELSE 0 END AS pts
        FROM matches WHERE FTR IS NOT NULL
        UNION ALL
        SELECT AwayTeam, FTAG, FTHG,
               CASE FTR WHEN 'A' THEN 3 WHEN 'D' THEN 1 ELSE 0 END
        FROM matches WHERE FTR IS NOT NULL
    )
    SELECT RANK() OVER (ORDER BY SUM(pts) DESC) AS rank,
           team,
           COUNT(*) AS played,
           SUM(pts) AS points,
           SUM(gf) - SUM(ga) AS goal_diff,
           ROUND(1.0 * SUM(pts) / COUNT(*), 2) AS pts_per_game
    FROM team_matches
    GROUP BY team
    HAVING COUNT(*) >= 100
    ORDER BY points DESC
    LIMIT 10
""",

"Model ratings vs last season points": """
    WITH recent AS (
        SELECT HomeTeam AS team,
               CASE FTR WHEN 'H' THEN 3 WHEN 'D' THEN 1 ELSE 0 END AS pts
        FROM matches WHERE Season = '2526' AND FTR IS NOT NULL
        UNION ALL
        SELECT AwayTeam,
               CASE FTR WHEN 'A' THEN 3 WHEN 'D' THEN 1 ELSE 0 END
        FROM matches WHERE Season = '2526' AND FTR IS NOT NULL
    ),
    season_table AS (
        SELECT team, SUM(pts) AS season_points FROM recent GROUP BY team
    )
    SELECT r.Rank AS elo_rank, r.Team, r.ELO, s.season_points
    FROM team_ratings r
    INNER JOIN season_table s ON r.Team = s.team
    ORDER BY r.ELO DESC
    LIMIT 10
""",

"Top goalscorers": """
    SELECT player, team, season,
           Performance_Gls AS goals,
           Performance_Ast AS assists,
           ROUND(Per_90_Minutes_Gls, 2) AS goals_per_90
    FROM players
    WHERE Playing_Time_90s >= 10
    ORDER BY Performance_Gls DESC
    LIMIT 10
""",

"Best home attacks (last season)": """
    SELECT HomeTeam AS team,
           ROUND(AVG(FTHG), 2) AS avg_goals_scored,
           ROUND(AVG(HS), 1) AS avg_shots,
           ROUND(AVG(HST), 1) AS avg_shots_on_target,
           ROUND(AVG("AS"), 1) AS avg_shots_allowed
    FROM matches
    WHERE Season = '2526'
    GROUP BY HomeTeam
    ORDER BY avg_goals_scored DESC
    LIMIT 8
""",
}


def main():
    if not os.path.exists(DB_FILE):
        print(f"Missing {DB_FILE}. Run: py src/build_database.py")
        return

    conn = sqlite3.connect(DB_FILE)
    pd.set_option("display.width", 200)

    for title, sql in QUERIES.items():
        print("=" * 70)
        print(title)
        print("=" * 70)
        try:
            df = pd.read_sql(sql, conn)
            print(df.to_string(index=False))
        except Exception as exc:
            print(f"Query failed: {exc}")
        print()

    conn.close()


if __name__ == "__main__":
    main()
