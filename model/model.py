# =====================================================
# ASSI Forecasting – FINAL BEST MODEL
# Adaptive Robust Ensemble + Safety Clipping
# =====================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------
# Paths
# ------------------------------------------
PROJECT_DIR = "."
MASTER_FILE = os.path.join(PROJECT_DIR, "outputs/master_monthly.csv")

SCORES_FILE = os.path.join(
    PROJECT_DIR, "outputs/tables/model_backtest_scores.csv"
)

FORECAST_FILE = os.path.join(
    PROJECT_DIR, "outputs/tables/forecast_top_hotspots_best_3m.csv"
)

TOP10_FILE = os.path.join(
    PROJECT_DIR, "outputs/tables/top10_predicted_hotspots_next_month.csv"
)

os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)
os.makedirs(os.path.dirname(FORECAST_FILE), exist_ok=True)
os.makedirs(os.path.dirname(TOP10_FILE), exist_ok=True)

# ------------------------------------------
# Load data
# ------------------------------------------
df = pd.read_csv(MASTER_FILE)
df["ds"] = pd.to_datetime(df["month_year"] + "-01")
df["y"] = df["assi"]

# ------------------------------------------
# Select top 20 hotspots
# ------------------------------------------
top_hotspots = (
    df.groupby(["state", "district", "pincode"])["y"]
      .mean()
      .sort_values(ascending=False)
      .head(20)
      .reset_index()
)

# ------------------------------------------
# SMAPE metric
# ------------------------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(
        np.abs(y_true - y_pred) /
        ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )

# =====================================================
# PART 1 — BACKTESTING
# =====================================================
results = []

for _, row in top_hotspots.iterrows():

    s, d, p = row["state"], row["district"], row["pincode"]

    ts = (
        df[(df.state == s) & (df.district == d) & (df.pincode == p)]
        [["ds", "y"]]
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if len(ts) < 8:
        continue

    test = ts.iloc[-2:]
    preds = []

    for i in range(len(test)):
        history = ts.iloc[:-(2 - i)]["y"]

        # ---- Adaptive window based on volatility
        cv = history.std() / (history.mean() + 1e-6)
        window = 5 if cv < 0.3 else 3
        hist = history.tail(window).values

        # ---- Ensemble components
        sma = hist.mean()
        wma = np.average(hist, weights=np.linspace(1, 2, len(hist)))
        med = np.median(hist)

        raw_pred = np.median([sma, wma, med])

        # ---- Safety clipping
        med6 = history.tail(6).median()
        lower = 0.7 * med6
        upper = 1.3 * med6
        final_pred = np.clip(raw_pred, lower, upper)

        preds.append(final_pred)

    smape_val = smape(test["y"].values, np.array(preds))

    results.append({
        "state": s,
        "district": d,
        "pincode": p,
        "smape_best_model": round(smape_val, 2)
    })

scores_df = pd.DataFrame(results)
scores_df.to_csv(SCORES_FILE, index=False)

print("Avg Best Model SMAPE:", round(scores_df.smape_best_model.mean(), 2))

# =====================================================
# PART 2 — FINAL FORECAST (NEXT 3 MONTHS)
# =====================================================
forecast_rows = []

for _, row in top_hotspots.iterrows():

    s, d, p = row["state"], row["district"], row["pincode"]

    ts = (
        df[(df.state == s) & (df.district == d) & (df.pincode == p)]
        [["ds", "y"]]
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if len(ts) < 6:
        continue

    history = ts["y"]
    cv = history.std() / (history.mean() + 1e-6)
    window = 5 if cv < 0.3 else 3
    hist = history.tail(window).values

    sma = hist.mean()
    wma = np.average(hist, weights=np.linspace(1, 2, len(hist)))
    med = np.median(hist)

    raw_pred = np.median([sma, wma, med])

    med6 = history.tail(6).median()
    lower = 0.7 * med6
    upper = 1.3 * med6
    final_pred = round(np.clip(raw_pred, lower, upper), 2)

    last_date = ts["ds"].iloc[-1]

    for i in range(1, 4):
        forecast_rows.append({
            "state": s,
            "district": d,
            "pincode": p,
            "forecast_month": (last_date + pd.DateOffset(months=i)).strftime("%Y-%m"),
            "yhat": final_pred
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(FORECAST_FILE, index=False)

# =====================================================
# PART 3 — TOP 10 HOTSPOTS NEXT MONTH
# =====================================================
next_month = forecast_df["forecast_month"].min()

top10_df = (
    forecast_df[forecast_df["forecast_month"] == next_month]
    .sort_values("yhat", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

top10_df.insert(0, "rank", range(1, len(top10_df) + 1))
top10_df.to_csv(TOP10_FILE, index=False)

print("Saved BEST forecasts to:", FORECAST_FILE)
print("Saved Top-10 predicted hotspots to:", TOP10_FILE)
