"""
Build the SQLite database
Loads the project CSV files into a single SQLite database so the data can be
queried with SQL instead of only pandas.

Output: data/football.db
"""

import os
import sqlite3
import pandas as pd

DB_FILE = os.path.join("data", "football.db")

TABLES = {
    "matches": "premier_league_matches.csv",
    "features": "features.csv",
    "players": "premier_league_players.csv",
    "team_ratings": "team_ratings.csv",
    "predictions": "predictions.csv",
}


def clean_columns(df):
    """SQL column names cannot contain spaces, %, + or start with a digit."""
    new_cols = []
    for c in df.columns:
        c = (str(c).strip()
             .replace("%", "_pct")
             .replace("+", "_plus_")
             .replace(" ", "_")
             .replace("-", "_")
             .replace("/", "_"))
        if c and c[0].isdigit():
            c = "_" + c
        new_cols.append(c)
    df.columns = new_cols
    return df


def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    print(f"Building {DB_FILE}\n")

    for table, filename in TABLES.items():
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            print(f"  {table:14s} skipped (no {filename})")
            continue
        df = pd.read_csv(path)
        df = clean_columns(df)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  {table:14s} {len(df):>6} rows, {len(df.columns)} columns")

    cur = conn.cursor()
    existing = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "matches" in existing:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(HomeTeam)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(AwayTeam)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(Season)")
    if "team_ratings" in existing:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ratings_team ON team_ratings(Team)")
    conn.commit()

    print(f"\nTables created: {', '.join(existing)}")
    conn.close()
    print("Done. Query it with: py src/run_queries.py")


if __name__ == "__main__":
    main()
