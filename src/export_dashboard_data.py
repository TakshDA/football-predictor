"""
Export Dashboard Data for Power BI
Writes data/team_ratings.csv and data/predictions.csv for the dashboard.
"""

import os
import pandas as pd
import joblib

MATCHES_FILE = os.path.join("data", "premier_league_matches.csv")
FIXTURES_FILE = os.path.join("data", "upcoming_fixtures.csv")
MODEL_FILE = os.path.join("models", "match_predictor.joblib")
RATINGS_OUT = os.path.join("data", "team_ratings.csv")
PRED_OUT = os.path.join("data", "predictions.csv")

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
    return m.drop_duplicates().dropna(subset=["Date", "FTR"]).sort_values("Date").reset_index(drop=True)


def current_state(matches):
    elos = {}
    for _, r in matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        eh, ea = elos.get(h, ELO_BASE), elos.get(a, ELO_BASE)
        exp_h = 1 / (1 + 10 ** ((ea - eh - ELO_HOME_ADV) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[r["FTR"]]
        elos[h] = eh + ELO_K * (s - exp_h)
        elos[a] = ea + ELO_K * ((1 - s) - (1 - exp_h))

    h = matches.copy(); h["team"] = h["HomeTeam"]
    h["gf"], h["ga"] = h["FTHG"], h["FTAG"]; h["pts"] = h["FTR"].map({"H": 3, "D": 1, "A": 0})
    a = matches.copy(); a["team"] = a["AwayTeam"]
    a["gf"], a["ga"] = a["FTAG"], a["FTHG"]; a["pts"] = a["FTR"].map({"H": 0, "D": 1, "A": 3})
    log = pd.concat([h[["Date", "team", "gf", "ga", "pts"]],
                     a[["Date", "team", "gf", "ga", "pts"]]], ignore_index=True).sort_values("Date")
    form = {}
    for team, g in log.groupby("team"):
        rec = g.tail(FORM_WINDOW)
        form[team] = {"form_pts": rec["pts"].mean(),
                      "form_gf": rec["gf"].mean(), "form_ga": rec["ga"].mean()}
    return elos, form


def export_ratings(elos, form):
    rows = [{"Team": t, "ELO": round(e, 1),
             "FormPoints": round(form.get(t, {}).get("form_pts", 0), 2),
             "AvgGoalsFor": round(form.get(t, {}).get("form_gf", 0), 2),
             "AvgGoalsAgainst": round(form.get(t, {}).get("form_ga", 0), 2)}
            for t, e in elos.items()]
    df = pd.DataFrame(rows).sort_values("ELO", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    df.to_csv(RATINGS_OUT, index=False)
    return df


def export_predictions(matches, elos, form):
    if not (os.path.exists(FIXTURES_FILE) and os.path.exists(MODEL_FILE)):
        print("  (skipping predictions: need upcoming_fixtures.csv and a trained model)")
        return None
    model = joblib.load(MODEL_FILE)["model"]
    fixtures = pd.read_csv(FIXTURES_FILE)
    league_avg = {"form_pts": 1.35, "form_gf": 1.4, "form_ga": 1.4}

    def feat(home, away):
        he, ae = elos.get(home, ELO_BASE), elos.get(away, ELO_BASE)
        hf, af = form.get(home, league_avg), form.get(away, league_avg)
        return {"home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                "home_form_gf": hf["form_gf"], "home_form_ga": hf["form_ga"],
                "home_form_sot_for": 4.5, "home_form_sot_against": 4.5, "home_form_pts": hf["form_pts"],
                "away_form_gf": af["form_gf"], "away_form_ga": af["form_ga"],
                "away_form_sot_for": 4.5, "away_form_sot_against": 4.5, "away_form_pts": af["form_pts"]}

    X = pd.DataFrame([feat(r["HomeTeam"], r["AwayTeam"]) for _, r in fixtures.iterrows()])[FEATURE_COLS]
    proba = model.predict_proba(X)
    cl = list(model.classes_)
    out = []
    for i, (_, r) in enumerate(fixtures.iterrows()):
        p = proba[i]
        out.append({"Home": r["HomeTeam"], "Away": r["AwayTeam"],
                    "HomeWin%": round(p[cl.index("H")] * 100, 1),
                    "Draw%": round(p[cl.index("D")] * 100, 1),
                    "AwayWin%": round(p[cl.index("A")] * 100, 1),
                    "Pick": {"H": r["HomeTeam"], "D": "Draw", "A": r["AwayTeam"]}[cl[p.argmax()]]})
    df = pd.DataFrame(out)
    df.to_csv(PRED_OUT, index=False)
    return df


def main():
    if not os.path.exists(MATCHES_FILE):
        print(f"Missing {MATCHES_FILE}. Run collect_data.py first.")
        return
    matches = load_matches()
    elos, form = current_state(matches)
    ratings = export_ratings(elos, form)
    print(f"Saved {len(ratings)} team ratings to {RATINGS_OUT}")
    print(ratings.head(5).to_string(index=False))
    preds = export_predictions(matches, elos, form)
    if preds is not None:
        print(f"Saved {len(preds)} predictions to {PRED_OUT}")


if __name__ == "__main__":
    main()
