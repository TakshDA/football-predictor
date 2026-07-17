"""
Stage 2 - Data Collection for the Premier League Match Predictor
Downloads historical Premier League match data from football-data.co.uk,
combines multiple seasons into one dataset, and saves it for modeling.
"""

import os
from io import StringIO

import pandas as pd
import requests

SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "premier_league_matches.csv")

KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]


def download_season(season):
    url = BASE_URL.format(season=season)
    print(f"  {season}: downloading...", end=" ")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["Season"] = season
        df = df.dropna(subset=["FTR"])
        print(f"{len(df)} matches")
        return df
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Collecting Premier League data from football-data.co.uk\n")

    frames = []
    for season in SEASONS:
        df = download_season(season)
        if df is not None:
            frames.append(df)

    if not frames:
        print("\nNo data downloaded. Check your internet connection and try again.")
        return

    all_data = pd.concat(frames, ignore_index=True)
    all_data["Date"] = pd.to_datetime(all_data["Date"], dayfirst=True, errors="coerce")
    all_data = all_data.sort_values("Date").reset_index(drop=True)
    all_data.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. {len(all_data)} matches saved to {OUTPUT_FILE}")
    print(f"Date range: {all_data['Date'].min().date()} to {all_data['Date'].max().date()}")
    print(f"Seasons collected: {all_data['Season'].nunique()}")


if __name__ == "__main__":
    main()
