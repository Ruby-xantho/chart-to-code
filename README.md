# Chart to Code

![PyPI version](https://img.shields.io/pypi/v/chart-to-code.svg)
[![Documentation Status](https://readthedocs.org/projects/chart-to-code/badge/?version=latest)](https://chart-to-code.readthedocs.io/en/latest/?version=latest)

Chart-to-code is a self-hosted trading assistant based on a trained vision-language-model and algorithmic trading concepts. 

* PyPI package: https://pypi.org/project/chart-to-code/
* Free software: MIT License
* Documentation: https://chart-to-code.readthedocs.io.

## About

# Chart-to-Code: VLM-Driven Trend Trader

A self-hosted, open-source toolkit to convert TradingView chart screenshots into executable PineScript strategies.
Leverages a vision-language model (VLM) to identify moving-average lines, crosses, and annotations, then auto-generates PineScript code. Ideal for traders, quant researchers, and anyone experimenting with algorithmic trading.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
   - [Requirements](#requirements)
   - [Installation](#installation)
4. [Usage](#usage)
   - [CLI Workflow](#cli-workflow)
   - [Docker (Optional)](#docker-optional)
5. [Examples](#examples)
6. [Folder Structure](#folder-structure)
7. [Configuration & Environment Variables](#configuration--environment-variables)
8. [Development & Contributing](#development--contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview

**Chart-to-Code** provides a reproducible pipeline that transforms static TradingView screenshots into ready-to-use PineScript strategy templates. The core idea is:

1. **Ingest** a screenshot of a TradingView chart containing moving-average lines (e.g., EMA50, EMA200).
2. **Run** a Vision-Language Model (e.g., qwen2.5-vl) to detect:
   - Symbol/ticker (e.g., “AAPL”, “BTCUSDT”)
   - Timeframe (e.g., “1H”, “4H”, “1D”)
   - Moving-average types & lengths (e.g., 50 EMA, 200 EMA)
   - Cross events (golden/death cross points)
   - Bullish/Bearish Trends

3. **Generate** PineScript code that sets up those MAs and defines buy/sell conditions based on the identified crosses.
4. **Output** a `.pine` file that can be imported or copy-pasted into TradingView’s Pine editor for immediate backtesting.

By chaining these steps into a single CLI/API, you get a fully local, open-source solution—no third-party cloud calls required.

---

## Key Features

- **VLM-Powered Chart Parsing**
  - Detect lines, labels, and crossover points directly from screenshots.
  - Identify instrument symbol and timeframe.
- **Automated PineScript Generation**
  - Generate valid PineScript v5 code with:
    - MA definitions (`ta.ema()`, `ta.sma()`, etc.)
    - Crossover logic (`ta.crossover()`, `ta.crossunder()`)
    - Basic plot styling (colors, line widths)
  - Optionally insert default stop-loss/take-profit placeholders.
- **Reproducible CLI Modules**
  - `ingest` → load and preprocess image.
  - `infer` → VLM inference + line extraction.
  - `generate` → PineScript code generation.
  - `export` → write `.pine` file and metadata manifest.
- **Self-Hostable**
  - Fully offline: runs on a single GPU (32–96 GB VRAM recommended).
  - Bundled Dockerfile for reproducibility.
  - vLLM engine
- **FAIR Compliance**
  - Automatically save metadata (image hash, model checkpoints, extracted features).
  - Export a `manifest.json` capturing inputs, VLM version, and generated code hash.

---

## Getting Started

### Requirements

- **Operating System:** Linux or macOS (tested on Ubuntu 20.04+).
- **GPU:** NVIDIA GPU with ≥32 GB VRAM (e.g., RTX PRO 6000, A100).
- **CUDA:** CUDA 12.x (compatible with PyTorch 2.x).
- **Python:** 3.10+
- **Disk Space:** ≥50 GB for model weights and indexing.
- **Memory:** ≥32 GB RAM.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/chart-to-code.git
   cd chart-to-code





## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
