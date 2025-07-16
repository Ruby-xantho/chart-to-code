# oscillator_plot.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO

def plot_oscillator(df, ao_fast=5, ao_slow=34) -> bytes:
    """
    Generate Awesome Oscillator histogram as PNG bytes.
    df: DataFrame with 'high' and 'low' columns and datetime index.
    ao_fast, ao_slow: window lengths for the fast and slow SMAs.
    Returns raw PNG bytes.
    """
    # Compute median price and moving averages
    hl2 = (df['high'] + df['low']) / 2
    sma_fast = hl2.rolling(window=ao_fast, min_periods=1).mean()
    sma_slow = hl2.rolling(window=ao_slow, min_periods=1).mean()
    ao = sma_fast - sma_slow

    # Difference for coloring
    diff = ao.diff()
    # Use PineScript’s official colors: teal for rising, red for falling
    colors = np.where(diff > 0, '#27c6da', '#9498a1')

    # Compute dynamic bar width: 80% of median time interval
    interval = df.index.to_series().diff().median()
    width = (interval / pd.Timedelta(days=1)) * 0.8

    # Create figure and axis
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot histogram bars
    ax.bar(
        df.index,
        ao,
        color=colors,
        width=width,
        align='center',
        antialiased=False
    )

    # Zero line
    ax.axhline(0, color='#888eb0', linewidth=1.0, linestyle='--', zorder=1)

    # Dark theme
    ax.set_facecolor('#131722')
    fig.patch.set_facecolor('#131722')
    ax.set_axisbelow(True)
    ax.grid(color='#2a2a2a', linewidth=0.4)

    # Hide x-axis ticks and labels
    ax.tick_params(axis='x', which='both', labelbottom=False, length=0)
    # Style y-axis ticks
    ax.tick_params(axis='y', which='both', colors='#888eb0')

    for spine in ax.spines.values():
        spine.set_color('#2a2a2a')

    # Remove x-axis label/title
    ax.xaxis.label.set_visible(False)

    # Label and limits, with extra room below zero
    ax.set_ylabel('AO', color='#888eb0', labelpad=8)
    y_max = ao.max()
    y_min = ao.min()
    span = y_max - y_min
    pad = span * 0.05
    bottom = y_min - pad * 2    # extra space below
    top = y_max + pad
    ax.set_ylim(bottom, top)

    # Save to buffer
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()
