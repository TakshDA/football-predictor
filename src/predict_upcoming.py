"""
Stage 5 - Predict Upcoming Fixtures
Uses the trained model plus each team's current ELO and recent form to
predict upcoming matches as probabilities H / D / A.

Needs data/upcoming_fixtures.csv with columns:
    HomeTeam,AwayTeam
Team names must match your data's spelling (e.g. "Man City", "Nott'm Forest").
"""

import os
import pandas as pd
import joblib

MATCHES_FILE = os.path.join("data", "premier_league_matches.csv")
FIXTURES_FILE = os.path.join("data", "upcoming_fixtures.csv")
MODEL_FILE = os.path.join("models", "match_predictor.joblib")

ELO_BASE, ELO_K, ELO_HOME_ADV, FORM_WINDOW = 1500, 20, 60, 5

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_gf", "home_form_ga", "home_form_sot_for",
    "home_form_sot_against", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_sot_for",
    "away_form_sot_against", "away_form_pts",
]


def load_matches():
    m = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    m = m.drop_duplicates().dropna(subset=["Date", "FTR"])
    return m.sort_values("Date").reset_index(drop=True)


def current_elo(matches):
    elos = {}
    for _, r in matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eh, ea = elos.get(h, ELO_BASE), elos.get(a, ELO_BASE)
        exp_h = 1 / (1 + 10 ** ((ea - eh - ELO_HOME_ADV) / 400))
        actual_h = {"H": 1.0, "D": 0.5, "A": 0.0}[r["FTR"]]
        elos[h] = eh + ELO_K * (actual_h - exp_h)
        elos[a] = ea + ELO_K * ((1 - actual_h) - (1 - exp_h))
    return elos


def current_form(matches):
    h = matches.copy()
    h["team"] = h["HomeTeam"]; h["gf"], h["ga"] = h["FTHG"], h["FTAG"]
    h["sot_for"], h["sot_against"] = h["HST"], h["AST"]
    h["pts"] = h["FTR"].map({"H": 3, "D": 1, "A": 0})
    a = matches.copy()
    a["team"] = a["AwayTeam"]; a["gf"], a["ga"] = a["FTAG"], a["FTHG"]
    a["sot_for"], a["sot_against"] = a["AST"], a["HST"]
    a["pts"] = a["FTR"].map({"H": 0, "D": 1, "A": 3})
    cols = ["Date", "team", "gf", "ga", "sot_for", "sot_against", "pts"]
    log = pd.concat([h[cols], a[cols]], ignore_index=True).sort_values("Date")

    stats = ["gf", "ga", "sot_for", "sot_against", "pts"]
    form = {}
    for team, g in log.groupby("team"):
        recent = g.tail(FORM_WINDOW)
        form[team] = {f"form_{s}": recent[s].mean() for s in stats}
    league_avg = {f"form_{s}": log[s].mean() for s in stats}
    return form, league_avg


def build_row(home, away, elos, form, league_avg):
    he, ae = elos.get(home, ELO_BASE), elos.get(away, ELO_BASE)
    hf = form.get(home, league_avg)
    af = form.get(away, league_avg)
    row = {"home_elo": he, "away_elo": ae, "elo_diff": he - ae}
    for s in ["gf", "ga", "sot_for", "sot_against", "pts"]:
        row[f"home_form_{s}"] = hf[f"form_{s}"]
        row[f"away_form_{s}"] = af[f"form_{s}"]
    return row


def main():
    for f in (MATCHES_FILE, FIXTURES_FILE, MODEL_FILE):
        if not os.path.exists(f):
            print(f"Missing {f}.")
            if f == FIXTURES_FILE:
                print("Create it with columns: HomeTeam,AwayTeam")
            return

    matches = load_matches()
    elos = current_elo(matches)
    form, league_avg = current_form(matches)
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    known = set(elos)

    fixtures = pd.read_csv(FIXTURES_FILE)
    rows = [build_row(r["HomeTeam"], r["AwayTeam"], elos, form, league_avg)
            for _, r in fixtures.iterrows()]
    X = pd.DataFrame(rows)[FEATURE_COLS]
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    print(f"{'Home':<14}{'Away':<14}{'Home%':>7}{'Draw%':>7}{'Away%':>7}  Pick")
    print("-" * 62)
    for i, (_, r) in enumerate(fixtures.iterrows()):
        p = proba[i]
        ph = p[classes.index("H")] * 100
        pd_ = p[classes.index("D")] * 100
        pa = p[classes.index("A")] * 100
        pick = {"H": r["HomeTeam"], "D": "Draw", "A": r["AwayTeam"]}[
            classes[p.argmax()]]
        warn = "" if r["HomeTeam"] in known and r["AwayTeam"] in known else "  (new team, low confidence)"
        print(f"{r['HomeTeam']:<14}{r['AwayTeam']:<14}"
              f"{ph:>6.0f}%{pd_:>6.0f}%{pa:>6.0f}%  {pick}{warn}")


if __name__ == "__main__":
    main()
