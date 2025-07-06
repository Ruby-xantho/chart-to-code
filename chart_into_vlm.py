import streamlit as st
import pandas as pd
import ccxt
import mplfinance as mpf
from io import BytesIO
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh
import base64
import time

# must be the very first Streamlit command to enable wide mode
st.set_page_config(page_title="Trading Assistant", layout="wide")

model_name = "/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2"

# sidebar for token selection via text input (max 10) with validation
exchange = ccxt.binance()
exchange.load_markets()
default_tickers = "BTC/USDT, ETH/USDT, NEAR/USDT"
tickers_input = st.sidebar.text_input(
    "Enter up to 10 tokens (comma-separated)", default_tickers
)
input_symbols = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(input_symbols) > 10:
    st.sidebar.error("Please enter at most 10 tickers.")
    st.stop()
invalid = [s for s in input_symbols if s not in exchange.symbols]
if invalid:
    st.sidebar.error(f"Invalid tickers: {', '.join(invalid)}")
    st.stop()
SYMBOLS = input_symbols

@st.cache_resource
def get_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8501/v1"
    )

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def plot_and_get_png(df: pd.DataFrame) -> bytes:
    # 1) compute three MAs
    df["SMMA14"] = df["close"].ewm(alpha=1/14, adjust=False).mean()
    df["EMA13"]  = df["close"].ewm(span=13, adjust=False).mean()
    df["EMA21"]  = df["close"].ewm(span=21, adjust=False).mean()

    # 2) build addplot objects
    apds = [
        mpf.make_addplot(df["SMMA14"], type="step",  color="#00bcd4", width=0.5),
        mpf.make_addplot(df["EMA13"],  type="line",  color="#673ab7", width=0.5),
        mpf.make_addplot(df["EMA21"],  type="line",  color="#056656", width=0.5),
    ]

    # 3) render to Figure of higher dpi
    fig, axes = mpf.plot(
        df, type="candle", style="charles", addplot=apds,
        returnfig=True, figsize=(6, 4), volume=False, tight_layout=True
    )
    
    ax = axes[0]
    
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def make_image_part(png_bytes: bytes):
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"}
    }

# title in the center
st.markdown("<h1 style='text-align: center;'>Trading Assistant</h1>", unsafe_allow_html=True)

# split symbols into rows of max 3
rows = [SYMBOLS[i:i+3] for i in range(0, len(SYMBOLS), 3)]

# auto-refresh every 5 minutes
_ = st_autorefresh(interval=300_000, key="ticker")

client = get_client()

for row in rows:
    cols = st.columns(len(row))
    for col, symbol in zip(cols, row):
        with col:
            st.subheader(symbol)
            df = fetch_ohlcv(symbol)
            png = plot_and_get_png(df)
            st.image(png, use_container_width=True)

            # Build the chat prompt
            system_msg = {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a market-chart analysis assistant."}
                ]
            }
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                        f"Here’s the latest {symbol} chart with SMMA(14), EMA(13), EMA(21): "
                        "Is the price above or below the trend? Choose between bullish or bearish or sideways."
                       
                    },
                    make_image_part(png)
                ]
            }

            start = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[system_msg, user_msg],
                max_tokens=128
            )
            latency = time.time() - start

            # capture and colorize the single-word response
            result = resp.choices[0].message.content.strip().lower()
            if "bullish" in result:
                display = f"<span style='color:green;font-weight:bold;'>{result}</span>"
            elif "bearish" in result:
                display = f"<span style='color:red;font-weight:bold;'>{result}</span>"
            else:
                display = result

            st.write(f"⏱️ {latency:.2f}s")
            st.markdown("**Analysis:**")
            st.markdown(display, unsafe_allow_html=True)
