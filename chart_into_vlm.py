import ccxt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

from streamlit_autorefresh import st_autorefresh
import time
import openai

exchange = ccxt.binance()
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=200):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df



def plot_candles(df):
    fig = go.Figure(
        data=[go.Candlestick(
            x=df.ts, open=df.open, high=df.high, low=df.low, close=df.close
        )]
    )
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=350)
    return fig



def fig_to_image_bytes(fig):
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf.getvalue()


st.title("Live BTC/USDT Chart")
placeholder = st.empty()

# Refresh every 120 seconds
count = st_autorefresh(interval=120_000, key="chart_timer")

# 1) Fetch data
df = fetch_ohlcv()
# 2) Plot
fig = plot_candles(df)
# 3) Display
placeholder.plotly_chart(fig)
# 4) Convert to bytes
img_bytes = fig_to_image_bytes(fig)
# 5) Call VLM with openai standard
response = openai.ChatCompletion.create(
    model="/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2",
    messages=[
        {"role":"user","content":"Analyze this latest BTC/USDT chart."}
    ],
    images=[{"data": img_bytes}]
)
st.write(response.choices[0].message["content"])
