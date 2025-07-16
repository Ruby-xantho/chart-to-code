# stock_rsi_plot.py
import matplotlib.pyplot as plt
from io import BytesIO

def plot_stock_rsi(df, timeperiod=14, fastk_period=3, fastd_period=3) -> bytes:
    """
    Generate Stochastic RSI plot based on the RSI of close prices without talib.
    Returns raw PNG bytes.
    """
    buf = BytesIO()
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/timeperiod, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/timeperiod, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Stochastic on RSI values
    min_rsi = rsi.rolling(window=timeperiod).min()
    max_rsi = rsi.rolling(window=timeperiod).max()
    fastk = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    fastd = fastk.rolling(window=fastd_period).mean()
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xticklabels([])
    ax.plot(df.index, fastk)
    ax.plot(df.index, fastd)
    #ax.plot(df.index, fastk, label='%K')
    #ax.plot(df.index, fastd, label='%D')
    ax.axhline(80, linestyle='--')
    ax.axhline(20, linestyle='--')
    #ax.set_title(f'Stoch RSI ({timeperiod},{fastk_period},{fastd_period})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(buf, format='png')

    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()
