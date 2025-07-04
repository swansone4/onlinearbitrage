# Tactical Arbitrage Engine

This tool performs intelligent Amazon FBA product selection by analyzing:
- Velocity (sales + rank + FBA comp.)
- Volatility (price changes)
- ROI and Profit
- Risk-adjusted product selection

## Features
- Upload CSVs from Tactical Arbitrage
- Adjust scoring weight presets
- Interactive charts (Plotly)
- Export buy list

## Run It Locally

```bash

cd ~
git clone https://github.com/[username]/onlinearbitrage.git
cd onlinearbitrage

# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
