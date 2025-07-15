import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Config
N_SAMPLES = 1000
OUT_DIR   = "synthetic_data"
os.makedirs(OUT_DIR, exist_ok=True)

entries = []

for i in range(N_SAMPLES):
    # 2) Simulate dates & candlestick OHLC
    dates = pd.date_range("2025-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100
    opens  = prices + np.random.randn(30)*0.5
    closes = prices + np.random.randn(30)*0.5
    highs  = np.maximum(opens, closes) + np.random.rand(30)
    lows   = np.minimum(opens, closes) - np.random.rand(30)

    df = pd.DataFrame({
        "open": opens, "high": highs,
        "low": lows,   "close": closes
    }, index=dates)

    # 3) Trend‑line fit on closing prices
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].values
    lr = LinearRegression().fit(X, y)
    trend_at_last = lr.predict([[len(df)-1]])[0]

    label = "above_trend" if df["close"].iat[-1] > trend_at_last else "below_trend"

    # 4) Plot & save
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(df.index, df["close"], label="Close")
    ax.plot(df.index, lr.predict(X), linestyle="--", label="Trend")
    ax.set_title(label)
    ax.legend()
    img_path = os.path.join(OUT_DIR, f"chart_{i:04d}.png")
    fig.savefig(img_path, dpi=100)
    plt.close(fig)

    # 5) Record JSON entry
    entries.append({
        "image": img_path,
        "label": label,
        "metadata": {
            "last_close": float(df["close"].iat[-1]),
            "trend_at_last": float(trend_at_last)
        }
    })

# 6) Dump manifest
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(entries, f, indent=2)
