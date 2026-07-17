"""
Stage 4 - Train the Match Predictor
Predicts Home/Draw/Away using a time-based split (train on past seasons,
test on the most recent). Saves the best model for predicting fixtures.
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

FEATURES_FILE = os.path.join("data", "features.csv")
MODEL_FILE = os.path.join("models", "match_predictor.joblib")

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_gf", "home_form_ga", "home_form_sot_for",
    "home_form_sot_against", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_sot_for",
    "away_form_sot_against", "away_form_pts",
]


def main():
    if not os.path.exists(FEATURES_FILE):
        print(f"Missing {FEATURES_FILE}. Run build_features.py first.")
        return
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(FEATURES_FILE, parse_dates=["Date"]).sort_values("Date")

    seasons = sorted(df["Season"].astype(str).unique())
    test_season = seasons[-1]
    train = df[df["Season"].astype(str) != test_season]
    test = df[df["Season"].astype(str) == test_season]

    X_train, y_train = train[FEATURE_COLS], train["FTR"]
    X_test, y_test = test[FEATURE_COLS], test["FTR"]

    print(f"Train: {len(train)} matches (seasons {seasons[0]}-{seasons[-2]})")
    print(f"Test:  {len(test)} matches (season {test_season})\n")

    baseline = y_test.value_counts(normalize=True).max()
    print(f"Baseline (always guess most common): {baseline:.1%}\n")

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42),
    }

    scored = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scored[name] = (model, acc)
        flag = "beats baseline" if acc > baseline else "below baseline"
        print(f"{name:22s} accuracy: {acc:.1%}  ({flag})")

    best_name = max(scored, key=lambda k: scored[k][1])
    best_model, best_acc = scored[best_name]
    print(f"\nBest model: {best_name} at {best_acc:.1%}\n")

    pred = best_model.predict(X_test)
    print("Detailed performance (best model):")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted) order [A, D, H]:")
    print(confusion_matrix(y_test, pred, labels=["A", "D", "H"]))

    if best_name == "Random Forest":
        imp = sorted(zip(FEATURE_COLS, best_model.feature_importances_),
                     key=lambda x: -x[1])
        print("\nTop predictive features:")
        for f, v in imp[:6]:
            print(f"  {f:24s} {v:.3f}")

    joblib.dump({"model": best_model, "features": FEATURE_COLS}, MODEL_FILE)
    print(f"\nSaved {best_name} to {MODEL_FILE}")


if __name__ == "__main__":
    main()
