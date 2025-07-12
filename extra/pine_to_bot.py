#!/usr/bin/env python3
"""
pine_to_bot.py

Read a PineScript file and generate a Python/ccxt trading‐bot skeleton.
Usage:
    python pine_to_bot.py path/to/strategy.pine > bot_strategy.py
"""

import re
import sys
from jinja2 import Template

# A simple Jinja2 template for Python bot
BOT_TEMPLATE = """
import ccxt
import pandas as pd
import time
import talib
from datetime import datetime, timezone

# ─── CONFIG ───────────────────────────────────────────────────────────────
API_KEY    = "<YOUR_BINANCE_API_KEY>"
API_SECRET = "<YOUR_BINANCE_SECRET>"
SYMBOL     = "{{ symbol }}"
TIMEFRAME  = "{{ timeframe }}"
POSITION_SIZE = 0.001

# Strategy parameters (from PineScript inputs)
{% for inp in inputs %}
{{ inp.name }} = {{ inp.default }}
{% endfor %}

# ─── SETUP ───────────────────────────────────────────────────────────────────
exchange = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True})
exchange.load_markets()

def fetch_ohlcv(symbol, since=None, limit=500):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
in_position = False
stop_loss  = None

while True:
    df = fetch_ohlcv(SYMBOL)
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values

    # Indicators
    {% for ind in indicators %}
    {{ ind.var }} = talib.{{ ind.fn }}({{ ind.args }})
    {% endfor %}

    # Signals
    cross_up   = {{ signals.open }}
    cross_down = {{ signals.close }}

    # Entry logic
    if cross_up and not in_position:
        stop_loss = float(df['low'][-1] - 2 * ATR[-1])
        exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
        in_position = True

    # Exit logic
    if in_position and (cross_down or close[-1] <= stop_loss):
        exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        in_position = False

    time.sleep(exchange.parse_timeframe(TIMEFRAME) * 60)
"""

def parse_pine(text):
    # 1) Symbol & timeframe from fetch_ohlcv call placeholder
    #    (we'll assume user will edit SYMBOL/TIMEFRAME manually after)
    symbol = "BTC/USDT"
    timeframe = "1h"

    # 2) inputs: input.int or input.float
    inputs = []
    for m in re.finditer(r'input\.(int|float)\s*\(\s*defval\s*=\s*([0-9\.]+)\s*,\s*title\s*=\s*"([^"]+)"', text):
        typ, default, title = m.groups()
        # turn title into a valid Python var name
        name = title.strip().lower().replace(' ', '_')
        inputs.append({'name': name, 'default': default})

    # 3) indicators: look for ta.XXX(...) assignments
    indicators = []
    for m in re.finditer(r'(\w+)\s*=\s*ta\.(\w+)\s*\(\s*([^)]*)\)', text):
        var, fn, args = m.groups()
        indicators.append({'var': var, 'fn': fn.upper() if fn.islower() else fn, 'args': args})

    # 4) signals: crossover & crossunder
    open_match  = re.search(r'open_trade\s*=\s*(ta\.crossover\([^)]*\))', text)
    close_match = re.search(r'close_trade\s*=\s*(ta\.crossunder\([^)]*\))', text)
    signals = {
        'open': open_match.group(1) if open_match else 'False',
        'close': close_match.group(1) if close_match else 'False'
    }

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'inputs': inputs,
        'indicators': indicators,
        'signals': signals
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python pine_to_bot.py strategy.pine", file=sys.stderr)
        sys.exit(1)

    pine_text = open(sys.argv[1], 'r').read()
    ctx = parse_pine(pine_text)
    bot_code = Template(BOT_TEMPLATE).render(**ctx)
    print(bot_code)

if __name__ == '__main__':
    main()
