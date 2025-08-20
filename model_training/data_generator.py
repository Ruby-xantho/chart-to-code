# data_generator.py
"""
Automates creation of a labeled chart‑analysis dataset by:

1. Iterating over a list of symbols and timeframes.
2. Fetching the latest 100 OHLC candles for each combination.
3. Generating three chart panels (price, Awesome Oscillator, Stochastic RSI) via:
     - main_plot.plot_main_chart
     - oscillator_plot.plot_oscillator
     - stock_rsi_plot.plot_stock_rsi
4. Applying rule_engine.evaluate_chart_logic to each DataFrame to produce:
     - a label
     - explanatory phrases
     - debug metadata (only for sample checking, not for training)
5. Saving each example into a JSON‑lines for training (and per‑sample JSON) with:
     - file paths to the rendered images
     - the generated label and supporting reasons
     - debug information (only in the per-sample JSON for sample checking)

The result is a large, rule‑annotated corpus of chart snapshots and human‑readable
commentary for model training.
"""

import os
import json
import time
import random
from itertools import product
import hashlib

import ccxt
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trading_assistant')))

from main_plot import plot_main_chart
from oscillator_plot import plot_oscillator
from stock_rsi_plot import plot_stock_rsi
from rule_engine import evaluate_chart_logic

# Configuration
BASE_DIR = "data"
FULL_DIR = os.path.join(BASE_DIR, "full")
TRAIN_DIR = os.path.join(BASE_DIR, "training")
IMAGE_DIRS = {
    "main": os.path.join(BASE_DIR, "panels/main"),
    "ao": os.path.join(BASE_DIR, "panels/ao"),
    "rsi": os.path.join(BASE_DIR, "panels/rsi")
}




SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT",
    "ADA/USDT", "SUI/USDT", "LINK/USDT", "HBAR/USDT", "AVAX/USDT", "LTC/USDT",
    "DOT/USDT", "NEAR/USDT", "MINA/USDT", "ALGO/USDT", "POL/USDT", "ARB/USDT",
    "SEI/USDT", "ATOM/USDT", "FIL/USDT", "FET/USDT", "OP/USDT", "TIA/USDT", "MANTA/USDT"
]

TIMEFRAMES = ["1h", "4h", "1d"]
MAX_EXAMPLES = 250

LABEL_QUOTA = {
    "Sell Signal": 50,
    "Possible Buy Entry": 50,
    "Bullish": 50,
    "Inconclusive": 50,
    "Bearish": 50
}

# Create directories
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
for path in IMAGE_DIRS.values():
    os.makedirs(path, exist_ok=True)

# Initialize exchange
exchange = ccxt.binance()
exchange.load_markets()

# Tracker
total_count = 0
collected = {label: 0 for label in LABEL_QUOTA}
seen_fingerprints = set()

# Main generation loop
while total_count < MAX_EXAMPLES:
    for symbol, timeframe in product(SYMBOLS, TIMEFRAMES):
        if total_count >= MAX_EXAMPLES:
            break

        try:
            # Fetch candles
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)

            # Deduplication fingerprint
            close_series = df["close"].round(3).astype(str)
            fingerprint = hashlib.md5(close_series.to_json().encode()).hexdigest()
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)


            # Generate charts (note unpack of rsi)
            main_img        = plot_main_chart(df)                # bytes
            ao_img          = plot_oscillator(df)                # bytes
            rsi_img, k_last, d_last = plot_stock_rsi(df)         # bytes, float, float

            # Evaluate rule-based logic
            label, reasoning, debug = evaluate_chart_logic(df)

            # Skip if quota for this label is full
            if collected[label] >= LABEL_QUOTA[label]:
                continue

            # Save image files
            id_str = f"{total_count:04d}"
            main_path = os.path.join(IMAGE_DIRS["main"], f"{id_str}_main.png")
            ao_path   = os.path.join(IMAGE_DIRS["ao"],   f"{id_str}_ao.png")
            rsi_path  = os.path.join(IMAGE_DIRS["rsi"],  f"{id_str}_rsi.png")

            with open(main_path, "wb") as f: f.write(main_img)
            with open(ao_path, "wb") as f: f.write(ao_img)
            with open(rsi_path, "wb") as f: f.write(rsi_img)

            # Save full JSON with debug info
            full_example = {
                "symbol": symbol,
                "timeframe": timeframe,
                "label": label,
                "reasoning": reasoning,
                "debug": debug,
                "images": {
                    "main": os.path.relpath(main_path, BASE_DIR),
                    "ao": os.path.relpath(ao_path, BASE_DIR),
                    "rsi": os.path.relpath(rsi_path, BASE_DIR)
                }
            }
            full_fname = f"{id_str}_{symbol.replace('/', '')}_{timeframe}.json"
            with open(os.path.join(FULL_DIR, full_fname), 'w') as f:
                json.dump(full_example, f, indent=2)

            # Save training-ready JSONL
            training_example = {
                "images": [
                    os.path.relpath(main_path, BASE_DIR),
                    os.path.relpath(ao_path, BASE_DIR),
                    os.path.relpath(rsi_path, BASE_DIR)
                ],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n<image>\n<image>\nWhat is the signal based on these charts?"
                    },
                    {
                        "from": "gpt",
                        "value": label + "\n- " + "\n- ".join(reasoning)
                    }
                ]
            }
            with open(os.path.join(TRAIN_DIR, "data.jsonl"), 'a') as f:
                f.write(json.dumps(training_example) + "\n")

            collected[label] += 1
            total_count += 1
            print(f"Saved [{label}] {total_count}/{MAX_EXAMPLES}: {full_fname}")

        except Exception as e:
            print(f"Error for {symbol} {timeframe}: {e}")
            continue

print(f"Finished generating {total_count} examples.")
