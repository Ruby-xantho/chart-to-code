import mplfinance as mpf
from io import BytesIO
import matplotlib.dates as mdates  

def plot_main_chart(df, style='charles', figsize=(6, 4), dpi=100) -> bytes:
    """
    Plot main candlestick chart with SMMA(14), EMA(13), EMA(21).
    Returns raw PNG bytes.
    """
    # Calculate moving averages
    df["SMMA14"] = df["close"].ewm(alpha=1/14, adjust=False).mean()
    df["EMA13"]  = df["close"].ewm(span=13, adjust=False).mean()
    df["EMA21"]  = df["close"].ewm(span=21, adjust=False).mean()

    # Build addplot objects
    apds = [
        mpf.make_addplot(df["SMMA14"], type="step",  color="#00bcd4", width=0.5),
        mpf.make_addplot(df["EMA13"],  type="line",  color="#673ab7", width=0.5),
        mpf.make_addplot(df["EMA21"],  type="line",  color="#056656", width=0.5),
    ]

    # Render to Figure
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=style,
        addplot=apds,
        returnfig=True,
        figsize=figsize,
        volume=False,
        tight_layout=True
    )

    ax = axes[0]

    ax.xaxis.set_tick_params(labelbottom=True)
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    # Format grid
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
