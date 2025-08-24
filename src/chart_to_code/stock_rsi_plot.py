# stock_rsi_plot.py
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

from chart_to_code.rule_engine import compute_stoch_rsi

def plot_stock_rsi(df: pd.DataFrame,
                   timeperiod: int = 14,
                   fastk_period: int = 14,
                   fastd_period: int = 3) -> tuple[bytes, float, float]:
    """
    Returns:
      - PNG bytes of the full %K and %D chart,
      - the last %K value,
      - the last %D value.
    """
    buf = BytesIO()

    # compute once with identical parameters
    fastk, fastd = compute_stoch_rsi(df['close'],
                                     rsi_period=timeperiod,
                                     stoch_period=fastk_period,
                                     smooth_k=fastd_period)

    # plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xticklabels([])
    ax.plot(df.index, fastk, label='%K')
    ax.plot(df.index, fastd, label='%D')
    ax.axhline(80, linestyle='--')
    ax.axhline(20, linestyle='--')
    ax.legend()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # extract last values
    k_last = float(fastk.iloc[-1])
    d_last = float(fastd.iloc[-1])
    return buf.getvalue(), k_last, d_last
