import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
import talib

# ─── CONFIG
API_KEY    = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_SECRET_KEY"
SYMBOL     = "BTC/USDT"           # e.g. "ETH/USDT", "NEAR/USDT", etc.
TIMEFRAME  = "1h"                 # must match your backtest timeframe
FAST_EMA   = 50
SLOW_EMA   = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
POSITION_SIZE = 0.001            # e.g. 0.001 BTC
# only trade inside this UTC window:
START_TS = datetime(2017,1,1, tzinfo=timezone.utc).timestamp()
END_TS   = datetime(2121,1,1, tzinfo=timezone.utc).timestamp()

# ─── SETUP
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})
exchange.load_markets()

def fetch_ohlcv(since=None, limit=500):
    data = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME,
                                since=since, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def within_window(ts):
    return START_TS <= ts.timestamp() <= END_TS

# ─── MAIN LOOP 
in_position = False
stop_loss  = None

while True:
    # 1) Pull the latest 500 bars
    df = fetch_ohlcv(limit=500)
    now_ts = df.index[-1]
    if not within_window(now_ts):
        print(f"> {now_ts} outside trading window")
        time.sleep(60)
        continue

    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values

    # 2) Indicators
    fast_ema = talib.EMA(close, FAST_EMA)
    slow_ema = talib.EMA(close, SLOW_EMA)
    atr      = talib.ATR(high, low, close, ATR_PERIOD)[-1]

    # 3) Signals
    cross_up   = fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]
    cross_down = fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]

    # 4) Manage entry
    if cross_up and not in_position:
        stop_loss = float(df['low'][-1] - 2 * atr)
        print(f"> Enter LONG @ {close[-1]:.2f}, SL @ {stop_loss:.2f}")
        order = exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
        in_position = True

    # 5) Manage exit
    # 5a) Stop-loss
    if in_position and close[-1] <= stop_loss:
        print(f"> Stop-Loss hit @ {close[-1]:.2f}")
        exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        in_position = False

    # 5b) EMA cross under
    elif in_position and cross_down:
        print(f"> EMA cross down @ {close[-1]:.2f}")
        exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        in_position = False

    # 6) Wait until next candle
    time.sleep(exchange.parse_timeframe(TIMEFRAME) * 60)
