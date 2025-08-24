# Chart to Code

![PyPI version](https://img.shields.io/pypi/v/chart-to-code.svg)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-hosted trading assistant powered by a trained vision-language model (VLM) for trend analysis and algorithmic trading decisions.

* PyPI package: https://pypi.org/project/chart-to-code/
* Repository: https://github.com/Ruby-xantho/chart-to-code
* Free software: MIT License
* Documentation: https://github.com/Ruby-xantho/chart-to-code#readme

## About

**Chart-to-Code** is a vision-language model trained specifically for financial chart analysis and trend trading. The system analyzes live market data, generates technical analysis charts, and provides trading signals through a trained VLM that understands price action, momentum indicators, and market trends.

## Key Features

- **Custom-Trained VLM**: Fine-tuned vision-language model specialized for financial chart analysis
- **Real-Time Market Analysis**: Live data fetching from cryptocurrency exchanges via CCXT
- **Multi-Panel Chart Generation**:
  - Price charts with moving averages (EMA13, EMA21, SMMA14)
  - Awesome Oscillator momentum analysis
  - Stochastic RSI indicators
- **Automated Trading Signals**: AI-generated buy/sell/hold recommendations
- **Rule-Based Validation**: Hybrid approach combining ML predictions with algorithmic rules
- **Interactive Web Interface**: Streamlit-based dashboard for real-time analysis
- **Exchange Integration Ready**: Framework prepared for future live trading connections

## Technical Architecture

The system combines several components:
- **VLM Core**: Custom-trained Qwen2.5-VL model for chart interpretation
- **Data Pipeline**: Real-time market data via CCXT (Binance, others)
- **Chart Generation**: Multi-panel technical analysis visualization
- **Signal Processing**: Rule engine for trade signal validation
- **Web Interface**: Streamlit app for user interaction

## Installation

### Basic Installation
```bash
git clone https://github.com/Ruby-xantho/chart-to-code.git
cd chart-to-code
pip install -e .
```

### If you have access to a NVIDIA GPU like an A40, you can try this
```bash
pip install -e ".[gpu]"
```

### Requirements
- **Python**: 3.10+
- **For local hosting the model and running it with VLM inference**: NVIDIA GPU with ≥32 GB VRAM (A40, A100, etc.)
- **Memory**: ≥4 GB RAM for basic usage, ≥32 GB for model training and self hosting the model

## Usage

### Run the Trading Assistant
```bash
streamlit run app/trading_assistant.py
```

The web interface allows you to:
1. Select cryptocurrency pairs for analysis
2. View real-time technical analysis charts
3. Get AI-powered trading signals
4. Monitor multiple assets simultaneously

### Supported Markets
Currently supports cryptocurrency markets via CCXT:
- Bitcoin (BTC/USDT)
- Ethereum (ETH/USDT)
- Major altcoins (SOL, ADA, LINK, etc.)

## Model Training

The VLM was trained on a custom dataset of:
- 1000+ labeled chart examples
- Multiple timeframes (1h, 4h, 1d)
- Five signal categories: Sell Signal, Possible Buy Entry, Bullish, Bearish, Inconclusive
- Rule-based ground truth labels for supervised learning

## Project Structure

```
chart-to-code/
├── src/chart_to_code/     # Core package
├── app/                   # Streamlit web application
├── model_training/        # VLM training pipeline
├── exchange_bots/         # Trading bot implementations
├── validate_model/        # Model validation tools
└── tests/                 # Test suite
```

## Future Development

- **Live Trading Integration**: Connect to exchange APIs for automated execution
- **Additional Markets**: Expand beyond crypto to forex, stocks, commodities  
- **Advanced Strategies**: Multi-timeframe analysis and complex trading logic
- **Portfolio Management**: Risk management and position sizing
- **Backtesting Engine**: Historical strategy performance analysis

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes. Trading involves significant risk of loss. Always do your own research and never invest more than you can afford to lose.

---

*This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.*
