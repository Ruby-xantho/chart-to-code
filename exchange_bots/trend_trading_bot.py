import os
import asyncio
import logging
import signal
import argparse
from typing import List

import pandas as pd
from ccxt.async_support import binance as AsyncBinance
from dotenv import load_dotenv


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a rotating file and console logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        "trend_bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


async def fetch_ohlcv(
    exchange: AsyncBinance,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100
) -> pd.DataFrame:
    """
    Fetch OHLCV data asynchronously and return a DataFrame.
    """
    raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend indicators."""
    df = df.copy()
    df["SMMA14"] = df["close"].ewm(alpha=1/14, adjust=False).mean()
    df["EMA13"] = df["close"].ewm(span=13, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()
    return df


def determine_trend(latest: pd.Series) -> str:
    """Return trend: bullish, bearish, or sideways."""
    close, smma, ema13, ema21 = (
        latest["close"], latest["SMMA14"], latest["EMA13"], latest["EMA21"]
    )
    if close > smma > ema13 > ema21:
        return "bullish"
    if close < smma < ema13 < ema21:
        return "bearish"
    return "sideways"


async def execute_trade(
    exchange: AsyncBinance,
    symbol: str,
    signal: str,
    amount: float,
    logger: logging.Logger
) -> None:
    """Place a market buy/sell based on signal."""
    try:
        if signal == "bullish":
            logger.info(f"BUY {symbol} @ market, amount={amount}")
            order = await exchange.create_market_buy_order(symbol, amount)
        elif signal == "bearish":
            logger.info(f"SELL {symbol} @ market, amount={amount}")
            order = await exchange.create_market_sell_order(symbol, amount)
        else:
            logger.debug(f"No trade for {symbol}, trend is sideways")
            return

        logger.info(f"Order executed: {order}")
    except Exception:
        logger.exception(f"Trade execution error for {symbol}")


class TrendBot:
    """Trend-based trading bot using CCXT and asyncio."""
    def __init__(
        self,
        symbols: List[str],
        api_key: str,
        api_secret: str,
        timeframe: str,
        amount: float,
        interval: int,
        live: bool,
        logger: logging.Logger
    ):
        load_dotenv()  # Load .env over OS env if present
        creds = {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
        self.exchange = AsyncBinance(creds)
        self.symbols = symbols
        self.timeframe = timeframe
        self.amount = amount
        self.interval = interval
        self.live = live
        self.logger = logger
        self._stopping = False

    async def start(self) -> None:
        """Begin the main loop."""
        self.logger.info(f"TrendBot starting for symbols: {self.symbols}")

        while not self._stopping:
            tasks = []
            for sym in self.symbols:
                tasks.append(self._process_symbol(sym))
            await asyncio.gather(*tasks)
            self.logger.debug(f"Sleeping for {self.interval}s...")
            await asyncio.sleep(self.interval)

    async def _process_symbol(self, symbol: str) -> None:
        """Fetch data, compute trend, optionally trade."""
        try:
            df = await fetch_ohlcv(self.exchange, symbol, self.timeframe)
            df = compute_indicators(df)
            latest = df.iloc[-1]
            trend = determine_trend(latest)
            self.logger.info(f"{symbol} trend: {trend}")

            if self.live:
                await execute_trade(self.exchange, symbol, trend, self.amount, self.logger)
        except Exception:
            self.logger.exception(f"Error processing {symbol}")

    def stop(self) -> None:
        self._stopping = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async trend-based trading bot for Binance"
    )
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--apikey", type=str, default=None)
    parser.add_argument("--secret", type=str, default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--amount", type=float, default=0.001)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--timeframe", type=str, default="1h")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    api_key = args.apikey or os.getenv("BINANCE_API_KEY")
    api_secret = args.secret or os.getenv("BINANCE_SECRET_KEY")

    logger = setup_logger()
    bot = TrendBot(
        symbols, api_key, api_secret,
        timeframe=args.timeframe,
        amount=args.amount,
        interval=args.interval,
        live=args.live,
        logger=logger
    )

    loop = asyncio.get_event_loop()

    # shutdown on SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    try:
        loop.run_until_complete(bot.start())
    finally:
        loop.run_until_complete(bot.exchange.close())
        loop.close()
        logger.info("TrendBot stopped")


if __name__ == "__main__":
    main()
