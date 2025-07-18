# rule_engine.py
import random
import numpy as np

# Templates for phrasing (expanded to 10 each)
trend_positive = [
    "Price is above the moving averages (bullish trend).",
    "Candlesticks are trading above trend lines.",
    "Market is holding above EMA and SMMA levels.",
    "Price continues to trade in an uptrend.",
    "The current close is comfortably above moving averages.",
    "Bullish structure confirmed above EMAs.",
    "Price remains elevated over trend levels.",
    "The uptrend remains intact.",
    "Market is above short- and long-term moving averages.",
    "Trend indicators support continued upside."
]

trend_negative = [
    "Price is below the moving averages (bearish trend).",
    "Candlesticks are under the trend lines.",
    "Market is trading beneath EMA and SMMA support.",
    "Downtrend evident with price under EMAs.",
    "Lower highs and lower lows dominate the structure.",
    "Bearish continuation below trend lines.",
    "Price action reflects weak momentum below trend.",
    "Market is under pressure beneath averages.",
    "Sustained price action below EMAs signals caution.",
    "Downward trend remains in control."
]

ao_positive = [
    "AO is positive, indicating bullish momentum.",
    "The Awesome Oscillator is above zero, suggesting strength.",
    "Momentum is positive according to AO.",
    "AO histogram is green and climbing.",
    "Positive AO supports upward price movement.",
    "AO remains in bullish territory.",
    "Rising AO bars indicate growing strength.",
    "Bullish momentum is building per AO.",
    "Zero-line cross confirms bullish shift in AO.",
    "AO confirms upward momentum bias."
]

ao_negative = [
    "AO is negative, indicating weakness.",
    "The Awesome Oscillator is below zero.",
    "Momentum has turned bearish according to AO.",
    "AO histogram shows consistent red bars.",
    "AO reflects declining market momentum.",
    "Bearish AO suggests downside risk.",
    "AO continues to print lower bars.",
    "Momentum remains weak as AO trends lower.",
    "Negative AO reading confirms lack of strength.",
    "AO drift into negative confirms bearish mood."
]

rsi_reset = [
    "Stoch RSI is under 25, suggesting a momentum reset.",
    "Stochastic RSI is oversold, indicating potential upside.",
    "RSI has fully reset and may support a bounce.",
    "Momentum cycle reset near bottom range.",
    "Stoch RSI signals a possible bottoming setup.",
    "Oversold conditions in RSI suggest rebound risk.",
    "Stoch RSI is deeply reset, preparing for reversal.",
    "Mean reversion may occur as RSI bottoms.",
    "Momentum exhaustion reached on downside.",
    "Conditions oversold based on both %K and %D."
]

rsi_normal = [
    "Stoch RSI is under 75, showing no overbought conditions.",
    "RSI remains in a healthy range.",
    "No oversold or overbought condition in RSI.",
    "Stochastic RSI values remain neutral.",
    "Neither overbought nor oversold signals present.",
    "RSI is balanced within expected range.",
    "Stoch RSI does not indicate excess.",
    "Momentum is contained within typical bounds.",
    "Mid-range RSI supports trend continuation.",
    "RSI is stable, not flashing reversal risk."
]

rsi_high = [
    "Stoch RSI is above 80, suggesting overbought conditions.",
    "RSI is elevated, warning of possible reversal.",
    "Momentum is stretched as RSI enters overbought zone.",
    "Stoch RSI in overbought range may trigger pullback.",
    "Both %K and %D above 80 — typical sell zone.",
    "Overbought conditions signal caution.",
    "RSI surge may lead to short-term exhaustion.",
    "Extreme RSI values could precede retracement.",
    "RSI at upper bound warns of market topping.",
    "Elevated momentum may invite sellers."
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

        # Stochastic RSI approximation
        rsi_period = 14
        rsi = compute_rsi(close, rsi_period)
        rsi_min = rsi.rolling(rsi_period).min()
        rsi_max = rsi.rolling(rsi_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)

        k = stoch_rsi.iloc[-1] * 100
        d = stoch_rsi.iloc[-3:].mean() * 100

        # Rule Logic
        if price_above_trend and price_above_trend_by_3_percent and k > 80 and d > 80:
            label = "Sell Signal"
            reasons = [
                random.choice(trend_positive),
                random.choice(rsi_high),
                "Price is significantly extended above trend (>3%)."
            ]

        elif price_above_trend and ao_positive_flag and k <= 25 and d <= 25:
            label = "Possible Buy Entry"
            reasons = [
                random.choice(trend_positive),
                random.choice(ao_positive),
                random.choice(rsi_reset)
            ]

        elif price_above_trend and ao_positive_flag and k < 75 and d < 75:
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

        elif not price_above_trend or not ao_positive_flag:
            label = "Inconclusive"
            reasons = [
                random.choice(trend_positive if price_above_trend else trend_negative),
                random.choice(ao_positive if ao_positive_flag else ao_negative),
                "Only partial alignment between price and momentum."
            ]

        else:
            label = "Inconclusive"
            reasons = ["Unclear signal based on chart data."]

        debug = {
            "price": round(float(latest_close), 2),
            "trend": round(float(trend), 2),
            "ao": round(float(ao_latest), 2),
            "%K": round(float(k), 2),
            "%D": round(float(d), 2)
        }

        return label, reasons, debug

    except Exception as e:
        return "Inconclusive", [f"Error evaluating logic: {str(e)}"], {}

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
