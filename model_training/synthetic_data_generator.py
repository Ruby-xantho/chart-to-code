# synthetic_data_generator.py
"""
Generates a synthetic dataset of chart-analysis examples by:
  1. Fetching 100 candles for each symbol and timeframe.
  2. Producing three image panels via the plotting modules:
     - main_plot.plot_main_chart
     - oscillator_plot.plot_oscillator
     - stock_rsi_plot.plot_stock_rsi
  3. Applying rule-based logic to label each example.
  4. Saving each example as JSON with base64-encoded images and label/reasoning.
"""

import json
import base64
import time
import random
from itertools import product

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
OUTPUT_DIR = "data/examples"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "NEAR/USDT"]
TIMEFRAMES = ["1h", "4h"]
MAX_EXAMPLES = 1000
LABEL_QUOTA = {
    "Sell Signal": 200,
    "Possible Buy Entry": 200,
    "Bullish": 200,
    "Inconclusive": 200,
    "Bearish": 200
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize exchange
exchange = ccxt.binance()
exchange.load_markets()

# Helper: encode image bytes to base64 string
def encode_img(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode('utf-8')

# Tracker
total_count = 0
collected = {label: 0 for label in LABEL_QUOTA}

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

            # Generate charts
            main_img = plot_main_chart(df)
            ao_img = plot_oscillator(df)
            rsi_img = plot_stock_rsi(df)

            # Evaluate rule-based logic
            label, reasoning = evaluate_chart_logic(df)

            # Skip if quota for this label is full
            if collected[label] >= LABEL_QUOTA[label]:
                continue

            # Save
            example = {
                "symbol": symbol,
                "timeframe": timeframe,
                "label": label,
                "reasoning": reasoning,
                "images": {
                    "main": encode_img(main_img),
                    "ao": encode_img(ao_img),
                    "rsi": encode_img(rsi_img)
                }
            }
            fname = f"{total_count:04d}_{symbol.replace('/', '')}_{timeframe}.json"
            with open(os.path.join(OUTPUT_DIR, fname), 'w') as f:
                json.dump(example, f, indent=2)

            collected[label] += 1
            total_count += 1
            print(f"Saved [{label}] {total_count}/{MAX_EXAMPLES}: {fname}")

        except Exception as e:
            print(f"Error for {symbol} {timeframe}: {e}")
            continue

print(f"Finished generating {total_count} examples.")
