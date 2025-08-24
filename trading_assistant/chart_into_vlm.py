import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import ccxt
import time
import base64
from openai import OpenAI


from main_plot import plot_main_chart
from oscillator_plot import plot_oscillator
from stock_rsi_plot import plot_stock_rsi
from utils import make_rows

# Import rule engine
from rule_engine import evaluate_chart_logic

# Streamlit page config
st.set_page_config(page_title="Trading Assistant", layout="wide")

model_name = "/trained_model/snapshots/files"

# Initialize exchange once and load markets
exchange = ccxt.binance()
exchange.load_markets()

# Load system and user prompts from files
with open("prompts/chart_analysis_system_prompt.md", "r") as f:
    system_prompt_text = f.read()
with open("prompts/prompt.txt", "r") as f:
    user_prompt_text = f.read()

# Sidebar form for input
with st.sidebar.form(key="settings_form"):
    st.markdown("### Settings")
    tickers_input = st.text_input(
        "Enter up to 10 tokens (comma-separated)",
        value="BTC/USDT, ETH/USDT, LINK/USDT"
    )
    run_button = st.form_submit_button(label="Run Analysis")

# Only run after user presses button
if not run_button:
    st.sidebar.info("Enter tickers (max 10) and click 'Run Analysis' to start.")
    st.stop()

# Process and validate input
input_symbols = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(input_symbols) > 10:
    st.sidebar.error("Please enter at most 10 tickers.")
    st.stop()
invalid = [s for s in input_symbols if s not in exchange.symbols]
if invalid:
    st.sidebar.error(f"Invalid tickers: {', '.join(invalid)}")
    st.stop()

SYMBOLS = input_symbols

# Auto-refresh every 5 minutes
_ = st_autorefresh(interval=300_000, key="ticker_refresh")

@st.cache_resource
def get_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://194.68.245.133:22184/v1"
    )

client = get_client()

# Helper to base64-encode image bytes for VLM
def make_image_part(png_bytes: bytes):
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

# Title
st.markdown("<h1 style='text-align: center;'>Trading Assistant</h1>", unsafe_allow_html=True)

# Split into rows of up to 3 symbols
rows = make_rows(SYMBOLS)

for row in rows:
    cols = st.columns(len(row))
    for col, symbol in zip(cols, row):
        if symbol is None:
            continue
        with col:
            st.subheader(symbol)
            # Fetch data
            data = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)

            # Plot charts
            main_png = plot_main_chart(df)
            osc_png  = plot_oscillator(df)
            # now returns (png_bytes, last_K, last_D)
            rsi_png, k_last, d_last = plot_stock_rsi(df)


            st.image(main_png, caption="Main Chart", use_container_width=True)
            st.image(osc_png, caption="Oscillator Panel", use_container_width=True)
            st.image(rsi_png, caption="Stochastic RSI Panel", use_container_width=True)

            # Evaluate rule engine for numeric debug values

            label, reasons, debug = evaluate_chart_logic(df)
            # override so they match the chart’s last points exactly:
            debug['%K'] = round(k_last, 2)
            debug['%D'] = round(d_last, 2)


            # Prepare debug text
            debug_text = (
                f"price: {debug.get('price')}, trend: {debug.get('trend')}, ao: {debug.get('ao')}, "
                f"%K: {debug.get('%K')}, %D: {debug.get('%D')}"
            )
            # Display debug values (for debugging)
            st.markdown(f"**Debug Values:** {debug_text}")

            # Build VLM prompt
            system_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]}
            user_msgs = []
            # Pass user prompt and debug data before images
            for img in [main_png, osc_png, rsi_png]:
                part = make_image_part(img)
                user_msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_text},
                        {"type": "text", "text": debug_text},
                        part
                    ]
                })

            # Call OpenAI once
            start = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[system_msg] + user_msgs,
                max_tokens=512
            )
            latency = time.time() - start

            # Display result with bold label
            result_text = resp.choices[0].message.content.strip()
            lines = result_text.split("\n", 1)
            label_line = lines[0]
            rest = lines[1] if len(lines) > 1 else ""

            st.write(f"⏱️ {latency:.2f}s")
            if rest:
                st.markdown(f"**{label_line}**\n{rest}")
            else:
                st.markdown(f"**{label_line}**")
