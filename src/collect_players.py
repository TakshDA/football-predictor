"""
Stage 2b - Player Stats Collection for the Premier League Match Predictor
Pulls season-level player statistics from FBref using soccerdata.
"""

import os

import pandas as pd
import soccerdata as sd

SEASONS = ["2324", "2425", "2526"]
LEAGUE = "ENG-Premier League"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "premier_league_players.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Collecting Premier League player stats from FBref")
    print("This is slow the first time. Please be patient.\n")

    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)

    print("Fetching standard player season stats...")
    players = fbref.read_player_season_stats(stat_type="standard")

    players = players.reset_index()
    players.columns = [
        "_".join(str(c) for c in col if c).strip("_")
        if isinstance(col, tuple) else str(col)
        for col in players.columns
    ]

    players.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. {len(players)} player-season rows saved to {OUTPUT_FILE}")
    print(f"Seasons: {sorted(SEASONS)}")
    print(f"Columns available: {len(players.columns)}")
    print("First few column names:")
    print("  " + ", ".join(list(players.columns)[:15]))


if __name__ == "__main__":
    main()
