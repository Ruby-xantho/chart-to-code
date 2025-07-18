# verify_dataset.py
import streamlit as st
import json
import os
from PIL import Image

# Configuration
EXAMPLES_DIR = "data/examples"
PANELS_DIR = "data/panels"

st.set_page_config(layout="wide")
st.title("📊 Dataset Verifier for Chart Signals")

# Sidebar
json_files = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json"))
selected_file = st.sidebar.selectbox("Select a sample:", json_files)

with open(os.path.join(EXAMPLES_DIR, selected_file)) as f:
    example = json.load(f)

# Metadata
st.subheader(f"{example['symbol']} — {example['timeframe']}")
st.markdown(f"**Label:** `{example['label']}`")

# Reasoning
with st.expander("Reasoning", expanded=True):
    for reason in example["reasoning"]:
        st.markdown(f"- {reason}")

# Debug (if available)
debug = example.get("debug")
if debug:
    with st.expander("Debug Values"):
        st.json(debug, expanded=False)

# Charts
col1, col2, col3 = st.columns(3)
with col1:
    st.image(os.path.join(PANELS_DIR, "main", os.path.basename(example['images']['main'])), caption="Main Chart")
with col2:
    st.image(os.path.join(PANELS_DIR, "ao", os.path.basename(example['images']['ao'])), caption="Awesome Oscillator")
with col3:
    st.image(os.path.join(PANELS_DIR, "rsi", os.path.basename(example['images']['rsi'])), caption="Stochastic RSI")
