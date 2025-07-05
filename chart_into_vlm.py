import base64
from io import BytesIO
import streamlit as st
import pandas as pd
import ccxt
import mplfinance as mpf
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

@st.cache_resource
def get_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8501/v1"
    )

def fetch_ohlcv():
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=200)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def plot_and_get_png(df):
    fig, _ = mpf.plot(
        df, type="candle", style="charles", returnfig=True, figsize=(8,4), volume=False
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def make_image_part(png_bytes: bytes):
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"}
    }

st.title("Trading Assistant")
placeholder = st.empty()

# auto-refresh every 5 minutes
_ = st_autorefresh(interval=300_000, key="ticker")

df = fetch_ohlcv()
png_bytes = plot_and_get_png(df)
placeholder.image(png_bytes, use_container_width=True)

client = get_client()

# build messages as a list of content‐part lists
system_msg = {
    "role": "system",
    "content": [
        {"type": "text", "text": (
            "You are a market‐chart analysis assistant. "
            "Read the following image and summarize key patterns."
        )}
    ]
}

user_msg = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Here is the latest BTC/USDT chart."},
        make_image_part(png_bytes)
    ]
}

resp = client.chat.completions.create(
    model="/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2", 
    messages=[system_msg, user_msg],
    max_tokens=512
)


reply = resp.choices[0].message

st.markdown("**Analysis:**")
st.write(reply.content)


