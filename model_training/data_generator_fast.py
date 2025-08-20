# data_generator_fast.py

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
BASE_DIR      = "data"
FULL_DIR      = os.path.join(BASE_DIR, "full")
TRAIN_DIR     = os.path.join(BASE_DIR, "training")
IMAGE_DIRS    = {
    "main": os.path.join(BASE_DIR, "panels/main"),
    "ao":   os.path.join(BASE_DIR, "panels/ao"),
    "rsi":  os.path.join(BASE_DIR, "panels/rsi"),
}

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT",
    "ADA/USDT", "SUI/USDT", "LINK/USDT", "HBAR/USDT", "AVAX/USDT", "LTC/USDT",
    "DOT/USDT", "NEAR/USDT", "MINA/USDT", "ALGO/USDT", "POL/USDT", "ARB/USDT",
    "SEI/USDT", "ATOM/USDT", "FIL/USDT", "FET/USDT", "OP/USDT", "TIA/USDT", "MANTA/USDT"
]

TIMEFRAMES    = ["1h", "2h", "4h", "8h", "12h", "1d"]
MAX_EXAMPLES  = 250

# Ensure dirs exist
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
for d in IMAGE_DIRS.values(): os.makedirs(d, exist_ok=True)

# Init exchange
exchange = ccxt.binance()
exchange.load_markets()

# Precompute & shuffle combos
combos = list(product(SYMBOLS, TIMEFRAMES))
random.shuffle(combos)

seen_fingerprints = set()
total_count = 0

for symbol, timeframe in combos:
    if total_count >= MAX_EXAMPLES:
        break

    try:
        # 1) Fetch candles
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df   = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)

        # 2) Dedupe
        fp = hashlib.md5(df["close"].round(3).astype(str).to_json().encode()).hexdigest()
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        # 3) Plot panels
        main_img = plot_main_chart(df)
        ao_img   = plot_oscillator(df)
        rsi_img, k_last, d_last = plot_stock_rsi(df)

        # 4) Label
        label, reasoning, debug = evaluate_chart_logic(df)

        # 5) Save files
        idx_str = f"{total_count:04d}"
        paths = {
            "main": os.path.join(IMAGE_DIRS["main"], f"{idx_str}_main.png"),
            "ao":   os.path.join(IMAGE_DIRS["ao"],   f"{idx_str}_ao.png"),
            "rsi":  os.path.join(IMAGE_DIRS["rsi"],  f"{idx_str}_rsi.png"),
        }
        for img, p in zip((main_img, ao_img, rsi_img), paths.values()):
            with open(p, "wb") as f: f.write(img)

        # 6) Write JSON with debug
        full = {
            "symbol": symbol,
            "timeframe": timeframe,
            "label": label,
            "reasoning": reasoning,
            "debug": debug,
            "images": {k: os.path.relpath(v, BASE_DIR) for k,v in paths.items()}
        }
        with open(os.path.join(FULL_DIR, f"{idx_str}_{symbol.replace('/','')}_{timeframe}.json"), "w") as f:
            json.dump(full, f, indent=2)

        # 7) Write JSONL for training
        train_example = {
            "images": list(os.path.relpath(v, BASE_DIR) for v in paths.values()),
            "conversations": [
                {"from":"human", "value":"<image>\n<image>\n<image>\nWhat is the signal based on these charts?"},
                {"from":"gpt", "value": label + "\n- " + "\n- ".join(reasoning)}
            ]
        }
        with open(os.path.join(TRAIN_DIR, "data.jsonl"), "a") as f:
            f.write(json.dumps(train_example) + "\n")

        total_count += 1
        print(f"[{total_count}/{MAX_EXAMPLES}] {symbol} {timeframe} → {label}")

    except Exception as e:
        print(f"Error on {symbol} {timeframe}: {e}")
        continue

print(f"Done: generated {total_count} examples.")
