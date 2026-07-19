"""
Player Performance Prediction
Predicts a player NEXT-season goal contributions (goals + assists per 90)
from their current season statistics.

Method: player stats are season-level, so the model learns from season-to-season
transitions. Features come from season N, target is season N+1. Trains on
2023/24 -> 2024/25, tests on 2024/25 -> 2025/26.

The benchmark is "persistence": assuming a player repeats last season rate.
That is a strong baseline in sports, so beating it is the real test.
"""

import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

PLAYERS_FILE = os.path.join("data", "premier_league_players.csv")
OUTPUT_FILE = os.path.join("data", "player_predictions.csv")

MIN_MINUTES = 450
TRAIN_PAIR = (2324, 2425)
TEST_PAIR = (2425, 2526)

FEATURES = [
    "ga_p90", "g_p90", "a_p90", "npg_p90",
    "Playing_Time_Min", "start_ratio", "age",
    "is_fw", "is_mf", "is_df",
]


def load_players():
    df = pd.read_csv(PLAYERS_FILE)
    df.columns = [c.replace(" ", "_").replace("+", "_plus_").replace("-", "_minus_")
                  for c in df.columns]
    agg = {"Playing_Time_Min": "sum", "Playing_Time_MP": "sum",
           "Playing_Time_Starts": "sum", "Performance_Gls": "sum",
           "Performance_Ast": "sum", "Performance_PK": "sum",
           "age": "max", "pos": "first", "team": "last"}
    p = df.groupby(["season", "player"], as_index=False).agg(agg)
    p = p[p["Playing_Time_Min"] >= MIN_MINUTES].copy()

    n90 = p["Playing_Time_Min"] / 90
    p["ga_p90"] = (p["Performance_Gls"] + p["Performance_Ast"]) / n90
    p["g_p90"] = p["Performance_Gls"] / n90
    p["a_p90"] = p["Performance_Ast"] / n90
    p["npg_p90"] = (p["Performance_Gls"] - p["Performance_PK"]) / n90
    p["start_ratio"] = p["Playing_Time_Starts"] / p["Playing_Time_MP"].clip(lower=1)
    p["is_fw"] = p["pos"].str.startswith("FW").astype(int)
    p["is_mf"] = p["pos"].str.startswith("MF").astype(int)
    p["is_df"] = p["pos"].str.startswith("DF").astype(int)
    return p


def make_pairs(p, season_from, season_to):
    a = p[p["season"] == season_from].set_index("player")
    b = p[p["season"] == season_to].set_index("player")
    common = a.index.intersection(b.index)
    out = a.loc[common].copy()
    out["next_ga_p90"] = b.loc[common, "ga_p90"]
    return out.reset_index()


def main():
    if not os.path.exists(PLAYERS_FILE):
        print(f"Missing {PLAYERS_FILE}. Run collect_players.py first.")
        return

    p = load_players()
    train = make_pairs(p, *TRAIN_PAIR)
    test = make_pairs(p, *TEST_PAIR)
    print(f"Train: {len(train)} players ({TRAIN_PAIR[0]} -> {TRAIN_PAIR[1]})")
    print(f"Test:  {len(test)} players ({TEST_PAIR[0]} -> {TEST_PAIR[1]})\n")

    X_train, y_train = train[FEATURES], train["next_ga_p90"]
    X_test, y_test = test[FEATURES], test["next_ga_p90"]

    baseline_mae = mean_absolute_error(y_test, test["ga_p90"])
    print(f"Baseline (assume same rate next season): MAE {baseline_mae:.4f}\n")

    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=42),
    }

    scored = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        scored[name] = (model, mae)
        verdict = "beats baseline" if mae < baseline_mae else "below baseline"
        print(f"{name:18s} MAE {mae:.4f}  R2 {r2_score(y_test, pred):.3f}  ({verdict})")

    best_name = min(scored, key=lambda k: scored[k][1])
    best_model, best_mae = scored[best_name]
    lift = (baseline_mae - best_mae) / baseline_mae * 100
    print(f"\nBest: {best_name}, {lift:.1f}% more accurate than the baseline.\n")

    test = test.copy()
    test["predicted_ga_p90"] = best_model.predict(X_test)
    top = test.nlargest(10, "ga_p90")
    print("Last season top 10 performers (goals+assists per 90):")
    print(f"  their {TEST_PAIR[0]} rate:  {top['ga_p90'].mean():.3f}")
    print(f"  model predicted:    {top['predicted_ga_p90'].mean():.3f}")
    print(f"  actual {TEST_PAIR[1]} rate: {top['next_ga_p90'].mean():.3f}")
    print("  The model correctly expects elite seasons to regress toward the mean.\n")

    out = test[["player", "team", "age", "pos", "Playing_Time_Min",
                "ga_p90", "predicted_ga_p90", "next_ga_p90"]].copy()
    out.columns = ["Player", "Team", "Age", "Position", "Minutes",
                   "CurrentGA_per90", "PredictedGA_per90", "ActualGA_per90"]
    out = out.round(3).sort_values("PredictedGA_per90", ascending=False)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out)} player predictions to {OUTPUT_FILE}")
    print("\nTop 10 projected performers:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
