# rule_engine.py
import random
import numpy as np

# Templates for phrasing (expandable to 5–10 each)
trend_positive = [
    "Price is above the moving averages (bullish trend).",
    "Candlesticks are trading above trend lines.",
    "Market is holding above EMA and SMMA levels."
]

trend_negative = [
    "Price is below the moving averages (bearish trend).",
    "Candlesticks are under the trend lines.",
    "Market is trading beneath EMA and SMMA support."
]


ao_positive = [
    "AO is positive, indicating bullish momentum.",
    "The Awesome Oscillator is above zero, suggesting strength.",
    "Momentum is positive according to AO."
]

ao_negative = [
    "AO is negative, indicating weakness.",
    "The Awesome Oscillator is below zero.",
    "Momentum has turned bearish according to AO."
]


rsi_reset = [
    "Stoch RSI is under 25, suggesting a momentum reset.",
    "Stochastic RSI is oversold, indicating potential upside.",
    "RSI has fully reset and may support a bounce."
]

rsi_normal = [
    "Stoch RSI is under 75, showing no overbought conditions.",
    "RSI remains in a healthy range.",
    "No oversold or overbought condition in RSI."
]

rsi_high = [
    "Stoch RSI is above 80, suggesting overbought conditions.",
    "RSI is elevated, warning of possible reversal.",
    "Momentum is stretched as RSI enters overbought zone."
]


def evaluate_chart_logic(df):
    try:
        close = df["close"]
        latest_close = close.iloc[-1]

        # Calculate moving averages
        ema13 = close.ewm(span=13, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        smma14 = close.ewm(alpha=1/14, adjust=False).mean()
        trend = (ema13.iloc[-1] + ema21.iloc[-1] + smma14.iloc[-1]) / 3

        price_above_trend = latest_close > trend
        price_above_trend_by_3_percent = (latest_close - trend) / trend >= 0.03

        # Awesome Oscillator (AO)
        median_price = (df['high'] + df['low']) / 2
        ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        ao_latest = ao.iloc[-1]
        ao_positive_flag = ao_latest > 0

        # Stochastic RSI approximation (not exact)
        rsi_period = 14
        rsi = compute_rsi(close, rsi_period)
        stoch_rsi = (rsi - rsi.rolling(rsi_period).min()) / (rsi.rolling(rsi_period).max() - rsi.rolling(rsi_period).min())
        stoch_rsi_value = stoch_rsi.iloc[-1] * 100

        # Rule Logic
        if price_above_trend and price_above_trend_by_3_percent and stoch_rsi_value > 80:
            label = "Sell Signal"
            reasons = [
                random.choice(trend_positive),
                random.choice(rsi_high),
                "Price is significantly extended above trend (>3%)."
            ]

        elif price_above_trend and ao_positive_flag and stoch_rsi_value <= 25:
            label = "Possible Buy Entry"
            reasons = [
                random.choice(trend_positive),
                random.choice(ao_positive),
                random.choice(rsi_reset)
            ]

        elif price_above_trend and ao_positive_flag and stoch_rsi_value < 75:
            label = "Bullish"
            reasons = [
                random.choice(trend_positive),
                random.choice(ao_positive),
                random.choice(rsi_normal)
            ]

        elif not price_above_trend and not ao_positive_flag:
            label = "Bearish"
            reasons = [
                random.choice(trend_negative),
                random.choice(ao_negative),
                "Both trend and momentum indicate weakness."
            ]

        else:
            label = "Inconclusive"
            reasons = [
                random.choice(trend_positive if price_above_trend else trend_negative),
                random.choice(ao_positive if ao_positive_flag else ao_negative),
                "Only partial alignment between price and momentum."
            ]

        return label, reasons

    except Exception as e:
        return "Inconclusive", [f"Error evaluating logic: {str(e)}"]


def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
