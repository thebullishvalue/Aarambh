# AARAMBH (आरंभ) - Residual Oscillator System

**A Hemrek Capital Product**

Multi-lookback breadth analysis for market top/bottom detection using fair value modeling and mean-reverting residual oscillators.

## Features

- **Fair Value Engine**: Ensemble regression (Ridge, Huber, OLS) for theoretical fair value calculation
- **Residual Oscillator**: Mean-reverting signal from price-value deviation
- **Multi-Lookback Breadth**: 5 timeframes (5D, 10D, 20D, 50D, 100D) for conviction scoring
- **Regime Detection**: Automatic oversold/overbought regime identification
- **Divergence Analysis**: Bullish and bearish divergence detection

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.10+
- streamlit
- pandas
- numpy
- plotly
- scipy
- scikit-learn
- statsmodels

## Usage

1. Load data via CSV upload or Google Sheets URL
2. Select target variable (e.g., NIFTY50_PE)
3. Select predictor variables (macro factors)
4. View multi-lookback analysis across all tabs

## Version

v1.1.0 - Hemrek Capital Design System
