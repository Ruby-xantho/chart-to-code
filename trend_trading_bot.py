import os
import time
import logging
import argparse

import pandas as pd
import ccxt

# Configure logging
def setup_logger(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", 
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Fetch OHLCV data
def fetch_ohlcv(exchange, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

# Compute trend indicators and simple trend logic
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMMA14"] = df["close"].ewm(alpha=1/14, adjust=False).mean()
    df["EMA13"]  = df["close"].ewm(span=13, adjust=False).mean()
    df["EMA21"]  = df["close"].ewm(span=21, adjust=False).mean()
    return df


def determine_trend(latest: pd.Series) -> str:
    close = latest["close"]
    smma = latest["SMMA14"]
    ema13 = latest["EMA13"]
    ema21 = latest["EMA21"]

    if close > smma and close > ema13 and close > ema21:
        return "bullish"
    if close < smma and close < ema13 and close < ema21:
        return "bearish"
    return "sideways"

# Execute trade based on signal
async def execute_trade(exchange, symbol: str, signal: str, amount: float):
    try:
        if signal == "bullish":
            logger.info(f"Placing BUY market order for {symbol}, amount={amount}")
            order = exchange.create_market_buy_order(symbol, amount)
        elif signal == "bearish":
            logger.info(f"Placing SELL market order for {symbol}, amount={amount}")
            order = exchange.create_market_sell_order(symbol, amount)
        else:
            logger.info(f"No trade for {symbol}, trend is sideways")
            return None
        logger.info(f"Order executed: {order}")
        return order

    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Trend-based trading bot for Binance")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT", 
                        help="Comma-separated list of symbols to trade")
    parser.add_argument("--apikey", type=str, default=None, help="Binance API key")
    parser.add_argument("--secret", type=str, default=None, help="Binance API secret")
    parser.add_argument("--trade", action="store_true", help="Enable live trading")
    parser.add_argument("--amount", type=float, default=0.001, help="Order amount per trade")
    parser.add_argument("--interval", type=int, default=3600, 
                        help="Polling interval in seconds (default 1h)")
    parser.add_argument("--timeframe", type=str, default="1h", help="OHLCV timeframe")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    api_key = args.apikey or os.getenv("BINANCE_API_KEY")
    api_secret = args.secret or os.getenv("BINANCE_SECRET_KEY")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True
    })

    logger.info(f"Starting trend bot for symbols: {symbols}")

    while True:
        for symbol in symbols:
            try:
                df = fetch_ohlcv(exchange, symbol, timeframe=args.timeframe)
                df = compute_indicators(df)
                latest = df.iloc[-1]
                signal = determine_trend(latest)
                logger.info(f"{symbol}: trend={signal}")

                if args.trade:
                    # async execution not strictly required for sync ccxt
                    execute_trade(exchange, symbol, signal, args.amount)

            except Exception as e:
                logger.error(f"Failed processing {symbol}: {e}")

        logger.info(f"Sleeping for {args.interval} seconds...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
