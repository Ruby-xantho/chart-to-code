import streamlit as st
import pandas as pd
import ccxt
import time
import base64
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

from main_plot import plot_main_chart
from oscillator_plot import plot_oscillator
from stock_rsi_plot import plot_stock_rsi
from utils import make_rows

# Streamlit page config
st.set_page_config(page_title="Trading Assistant", layout="wide")

model_name = "/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2"

# Initialize exchange once and load markets
exchange = ccxt.binance()
exchange.load_markets()

# Sidebar inputs
tickers_input = st.sidebar.text_input(
    "Enter up to 10 tokens (comma-separated)",
    "BTC/USDT, ETH/USDT, NEAR/USDT"
)
input_symbols = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
# Validate input count and symbols
if len(input_symbols) > 10:
    st.sidebar.error("Please enter at most 10 tickers.")
    st.stop()
invalid = [s for s in input_symbols if s not in exchange.symbols]
if invalid:
    st.sidebar.error(f"Invalid tickers: {', '.join(invalid)}")
    st.stop()
SYMBOLS = input_symbols

osc_kind = st.sidebar.selectbox("Oscillator", ["ao", "rsi"])

# auto-refresh every 5 minutes
_ = st_autorefresh(interval=300_000, key="ticker")

@st.cache_resource
def get_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8501/v1"
    )

client = get_client()

# Helper to base64-encode image bytes for VLM
def make_image_part(png_bytes: bytes):
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

# Title
st.markdown("<h1 style='text-align: center;'>Trading Assistant</h1>", unsafe_allow_html=True)

# Split into rows of 2 or 3
rows = make_rows(SYMBOLS)

for row in rows:
    cols = st.columns(len(row))
    for col, symbol in zip(cols, row):
        with col:
            if symbol is None:
                st.empty()
                continue

            st.subheader(symbol)
            # Fetch OHLCV from initialized exchange
            data = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)

            # Generate images using modular functions
            main_png = plot_main_chart(df)
            osc_png  = plot_oscillator(df)
            rsi_png  = plot_stock_rsi(df)

            # Display charts
            st.image(main_png, caption="Main Chart", use_container_width=True)
            st.image(osc_png,  caption="Oscillator Panel", use_container_width=True)
            st.image(rsi_png,  caption="Stochastic RSI Panel", use_container_width=True)

            # Build VLM prompt with three images
            system_msg = {"role": "system", "content": [{"type": "text", "text": "You are a market-chart analysis assistant."}]}
            user_msgs = [
                {"role": "user", "content": [{"type": "text", "text": f"Here is {symbol} main chart."}]},
                {"role": "user", "content": [{"type": "text", "text": "Oscillator panel."}]},
                {"role": "user", "content": [{"type": "text", "text": "Stochastic RSI panel."}]}
            ]
            user_msgs[0]["content"].append(make_image_part(main_png))
            user_msgs[1]["content"].append(make_image_part(osc_png))
            user_msgs[2]["content"].append(make_image_part(rsi_png))

            start = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[system_msg] + user_msgs,
                max_tokens=512
            )
            latency = time.time() - start

            # Colorize
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
