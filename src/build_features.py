"""
Stage 3 - Feature Engineering for the Premier League Match Predictor
Turns raw match results into leakage-free predictive features.
Features merge back on a unique match_id (never on date), so the join
is provably one-to-one and can never explode.
"""

import os
import pandas as pd

INPUT_FILE = os.path.join("data", "premier_league_matches.csv")
OUTPUT_FILE = os.path.join("data", "features.csv")

FORM_WINDOW = 5
ELO_K = 20
ELO_HOME_ADV = 60
ELO_BASE = 1500


def add_elo(matches):
    elos = {}
    home_elo, away_elo = [], []
    for _, r in matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eh = elos.get(h, ELO_BASE)
        ea = elos.get(a, ELO_BASE)
        home_elo.append(eh)
        away_elo.append(ea)
        exp_h = 1 / (1 + 10 ** ((ea - eh - ELO_HOME_ADV) / 400))
        actual_h = {"H": 1.0, "D": 0.5, "A": 0.0}[r["FTR"]]
        elos[h] = eh + ELO_K * (actual_h - exp_h)
        elos[a] = ea + ELO_K * ((1 - actual_h) - (1 - exp_h))
    matches = matches.copy()
    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    return matches


def to_team_log(matches):
    home = matches.copy()
    home["venue"] = "home"
    home["team"] = home["HomeTeam"]
    home["gf"], home["ga"] = home["FTHG"], home["FTAG"]
    home["sot_for"], home["sot_against"] = home["HST"], home["AST"]
    home["pts"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})

    away = matches.copy()
    away["venue"] = "away"
    away["team"] = away["AwayTeam"]
    away["gf"], away["ga"] = away["FTAG"], away["FTHG"]
    away["sot_for"], away["sot_against"] = away["AST"], away["HST"]
    away["pts"] = away["FTR"].map({"H": 0, "D": 1, "A": 3})

    cols = ["match_id", "Date", "team", "venue",
            "gf", "ga", "sot_for", "sot_against", "pts"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    return log.sort_values(["Date", "match_id"]).reset_index(drop=True)


def add_rolling_form(log):
    log = log.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    grp = log.groupby("team")
    stats = ["gf", "ga", "sot_for", "sot_against", "pts"]
    for col in stats:
        log[f"form_{col}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
        )
    return log[["match_id", "venue"] + [f"form_{c}" for c in stats]]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE}. Run collect_data.py first.")
        return

    matches = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    n_raw = len(matches)
    dupes = matches.duplicated().sum()
    matches = matches.drop_duplicates()
    n_bad_date = matches["Date"].isna().sum()
    matches = matches.dropna(subset=["Date", "FTR"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    matches["match_id"] = range(len(matches))

    print(f"Loaded {n_raw} rows.")
    if dupes:
        print(f"  Removed {dupes} exact duplicate rows.")
    if n_bad_date:
        print(f"  Removed {n_bad_date} rows with an unreadable/blank date.")
    print(f"Using {len(matches)} clean matches.\n")

    matches = add_elo(matches)

    form = add_rolling_form(to_team_log(matches))
    home_form = (form[form["venue"] == "home"]
                 .drop(columns="venue").add_prefix("home_")
                 .rename(columns={"home_match_id": "match_id"}))
    away_form = (form[form["venue"] == "away"]
                 .drop(columns="venue").add_prefix("away_")
                 .rename(columns={"away_match_id": "match_id"}))

    matches = matches.merge(home_form, on="match_id", how="left")
    matches = matches.merge(away_form, on="match_id", how="left")

    assert len(matches) == matches["match_id"].nunique(), "Merge exploded rows!"

    feature_cols = [
        "home_elo", "away_elo", "elo_diff",
        "home_form_gf", "home_form_ga", "home_form_sot_for",
        "home_form_sot_against", "home_form_pts",
        "away_form_gf", "away_form_ga", "away_form_sot_for",
        "away_form_sot_against", "away_form_pts",
    ]
    out = matches[["Date", "Season", "HomeTeam", "AwayTeam"] + feature_cols + ["FTR"]]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. {len(out)} matches with full features saved to {OUTPUT_FILE}")
    print(f"Features per match: {len(feature_cols)}")
    dist = out["FTR"].value_counts()
    print(f"Target distribution (H/D/A): {dist.to_dict()}")
    print(f"Baseline to beat (always guess most common): {dist.max() / len(out):.1%}")

if __name__ == "__main__":
    main()
