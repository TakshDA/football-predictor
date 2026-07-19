## Player performance prediction

A second model predicts individual player output: **goals + assists per 90
minutes for the following season**, based on current-season statistics.

Because player data is season-level, the model learns from season-to-season
transitions. It trains on the 2023/24 to 2024/25 transition and is tested on
2024/25 to 2025/26.

The benchmark is *persistence*, simply assuming a player repeats last season's
rate. That is a strong baseline in sports, which makes it a fair test.

| Model | MAE | vs baseline |
|-------|-----|-------------|
| Ridge Regression | **0.100** | 17.9% better |
| Random Forest | 0.105 | 14.3% better |
| Baseline (persistence) | 0.122 | — |

### Regression to the mean

The most interesting result is what the model learned about elite seasons.
Taking the ten highest-scoring players of 2024/25:

| | Goals + assists per 90 |
|---|---|
| Their 2024/25 rate | 0.885 |
| Model prediction | 0.718 |
| Actual 2025/26 rate | 0.528 |

The model correctly anticipates that outlier seasons regress toward the mean,
rather than naively projecting them forward. Mohamed Salah's 1.255 was
projected down to 0.875 (actual: 0.587). Erling Haaland is a notable exception,
projected at 0.706 but actually improving to 1.067, a reminder that genuine
outliers exist.
