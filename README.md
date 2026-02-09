# Stock Price Modeling Using AR & MA Models

## Project Overview
This project demonstrates the development and implementation of Autoregressive (AR) and Moving Average (MA) models to analyze and forecast short-term stock price behavior. It showcases how to leverage historical price values, account for past forecast errors, and evaluate model quality using advanced statistical diagnostics.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Key Features](#key-features)
5. [Installation and Setup](#installation--setup)
6. [Dataset Information](#dataset-information)
7. [Model Implementation](#model-implementation)
8. [Results and Findings](#results--findings)
9. [Usage Guide](#usage-guide)
10. [Limitations and Future Improvements](#limitations--future-improvements)

---

## Problem Statement

Traditional trend-based analysis fails to capture **temporal dependencies** in financial time series data. The project addresses:

- **Challenge**: Improper lag selection results in overfitting or weak predictive performance
- **Solution**: Develop AR, MA, and ARMA models with optimal parameter selection using AIC/BIC criteria
- **Goal**: Provide reliable diagnostic measures and accurate short-term stock price forecasts

---

## System Requirements

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| Pandas | Latest |
| NumPy | Latest |
| Matplotlib | Latest |
| Statsmodels | Latest |
| Scikit-learn | Latest |
| yfinance | Latest |
| SciPy | Latest |

---

## Project Structure

```
NLP Project/
├── NLP.ipynb                 # Main notebook with all implementations
├── README.md                 # Project documentation (this file)
└── Questions.md              # Q&A and learning resources
```

---

## Key Features

### 1. Data Preparation
- Real-time stock data fetching using yfinance
- Closing price extraction and visualization
- Differencing for stationarity transformation

### 2. Stationarity Testing
- Augmented Dickey-Fuller (ADF) test
- Pre and post-differencing analysis
- Statistical validation (p-value < 0.05)

### 3. Lag Selection
- Autocorrelation Function (ACF) plots
- Partial Autocorrelation Function (PACF) plots
- Optimal lag identification for AR and MA components

### 4. Model Development
- AR(p) Model: Autoregressive component
- MA(q) Model: Moving Average component
- ARMA(p,q) Model: Combined approach
- Grid search for optimal parameters (p, q)

### 5. Comprehensive Diagnostics
- Standardized residuals analysis
- Histogram and Kernel Density Estimation (KDE)
- Q-Q plots for normality assessment
- ACF of residuals for independence check
- Residual statistical measures (Mean, Std Dev, Skewness, Kurtosis)

### 6. Model Comparison
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Log-Likelihood values
- Performance metrics ranking

### 7. Forecasting
- 30-day ahead predictions
- 95% confidence intervals
- Visual representation with uncertainty bands

### 8. Multi-Stock Analysis
- Robustness testing across AAPL, MSFT, GOOGL
- Comparative performance analysis
- Model generalization verification

---

## Installation and Setup

### Step 1: Install Required Packages
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn yfinance scipy
```

### Step 2: Launch Jupyter Notebook
```bash
jupyter notebook NLP.ipynb
```

### Step 3: Execute Cells Sequentially
Run each cell from top to bottom using Shift + Enter

---

## Dataset Information

### Data Source
Yahoo Finance (yfinance) - Real-time and Historical Stock Data

### Parameters
| Parameter | Value |
|-----------|-------|
| Stock Ticker | AAPL (Apple Inc.) |
| Start Date | 2019-01-01 |
| End Date | 2024-01-01 |
| Time Period | 5 years (1260 trading days) |
| Data Type | Daily closing prices |

### Multi-Stock Testing
- AAPL: Apple Inc.
- MSFT: Microsoft Corporation
- GOOGL: Alphabet Inc. (Google)

```python
# Example: Change ticker for different stocks
ticker = "MSFT"  # or "GOOGL", "TSLA", etc.
data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
```

---

## Model Implementation

### Cell 1: Dependencies and Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
```

### Cell 2: Data Download and Visualization
Fetches historical stock prices, creates price trend plot, and displays summary statistics.

### Cell 3: Stationarity Testing
Applies ADF test to original series to determine stationarity requirements.

### Cell 4: Differencing and Re-testing
Applies first difference transformation and validates stationarity with ADF test.

### Cell 5: ACF and PACF Analysis
Generates ACF and PACF plots to identify optimal AR and MA components.

### Cell 6-8: AR, MA, ARMA Models
```python
# AR(2) Model
ar_model = ARIMA(diff_price, order=(2, 0, 0))
ar_result = ar_model.fit()

# MA(2) Model
ma_model = ARIMA(diff_price, order=(0, 0, 2))
ma_result = ma_model.fit()

# ARMA(2,2) Model
arma_model = ARIMA(diff_price, order=(2, 0, 2))
arma_result = arma_model.fit()
```

### Cell 9: Grid Search Optimization
Evaluates 16 different model configurations and selects the best model based on AIC criterion.

### Cell 10: Diagnostic Visualizations
Generates 2x2 subplot with residual analysis, normality assessment, and ACF plots.

### Cell 11: Forecasting
Generates 30-step ahead predictions with 95% confidence intervals.

### Cell 12: Model Comparison
Compares all models using AIC, BIC, and Log-Likelihood metrics.

### Cell 13: Insights and Interpretation
Provides stationarity results, optimal lag values, and residual diagnostics.

---

## Results and Findings

### Best Model Performance
| Metric | Value |
|--------|-------|
| Model | ARIMA(2, 0, 3) |
| AIC | 5709.81 |
| BIC | 5745.76 |
| Log-Likelihood | -2847.90 |

### Model Comparison
| Model | AIC | BIC | Status |
|-------|-----|-----|--------|
| AR(2) | 5711.35 | 5731.90 | Baseline |
| MA(2) | 5711.26 | 5731.81 | Close |
| ARMA(2,2) | 5715.16 | 5745.98 | Worse |
| ARIMA(2,0,3) | 5709.81 | 5745.76 | Best |

### Diagnostic Statistics
- Mean of Residuals: 0.004 (close to zero)
- Standard Deviation: 2.33 (consistent volatility)
- Skewness: -0.12 (nearly symmetric)
- Kurtosis: 2.50 (near normal distribution)

---

## Usage Guide

### Basic Usage
```python
# 1. Import and download data
ticker = "AAPL"
data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
stock_price = data['Close']

# 2. Apply differencing for stationarity
diff_price = stock_price.diff().dropna()

# 3. Fit optimal model
model = ARIMA(diff_price, order=(2, 0, 3))
result = model.fit()

# 4. Make forecast
forecast = result.get_forecast(steps=30)
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# 5. Display results
print(result.summary())
```

### Modify for Different Stocks
```python
# Example: Tesla stock
ticker = "TSLA"
# Rest of code remains the same

# Example: Amazon stock
ticker = "AMZN"
# Rest of code remains the same
```

### Adjust Forecast Horizon
```python
# 5-day forecast
forecast = result.get_forecast(steps=5)

# 60-day forecast
forecast = result.get_forecast(steps=60)
```

---

## Limitations and Future Improvements

### Current Limitations
1. Linear Assumption: Model assumes linear relationships; may miss regime shifts
2. External Factors: News, earnings, market events not incorporated
3. Horizon Constraint: Best for 1-30 day forecasts; diverges for longer periods
4. Seasonality: No seasonal component (SARIMA not implemented)
5. Single Series: No multivariate relationships considered

### Future Enhancement Opportunities
1. SARIMA Models: Incorporate seasonal patterns
2. Ensemble Methods: Combine multiple model predictions
3. Machine Learning: LSTM/GRU for non-linear patterns
4. Exogenous Variables: Include market indicators, volume, sentiment
5. Real-time Updates: Automated model retraining
6. Risk Metrics: Value-at-Risk (VaR) calculation
7. Multi-step Validation: Walk-forward cross-validation
8. Trading System: Integrate with backtesting framework

---

## Key Concepts and Formulas

### AR Model (Autoregressive)
X_t = c + phi_1*X_(t-1) + phi_2*X_(t-2) + ... + phi_p*X_(t-p) + epsilon_t

### MA Model (Moving Average)
X_t = mu + epsilon_t + theta_1*epsilon_(t-1) + theta_2*epsilon_(t-2) + ... + theta_q*epsilon_(t-q)

### ARIMA Model
Delta^d*X_t = c + sum(phi_i*Delta^d*X_(t-i)) + sum(theta_j*epsilon_(t-j)) + epsilon_t

### AIC Criterion
AIC = 2k - 2*ln(L)
where k = number of parameters, L = maximum likelihood

---

## Practical Applications

### 1. Short-term Trading (5-10 days)
- Support buy/sell signals
- Detect momentum changes
- Validate technical patterns

### 2. Portfolio Management
- Identify volatility periods
- Allocate based on forecast confidence
- Rebalance timing optimization

### 3. Risk Management
- Estimate price uncertainty
- Set stop-loss levels
- Calculate hedge ratios

### 4. Investment Alerts
- Deviation detection
- Threshold-based notifications
- Anomaly identification

### 5. Strategic Planning
- Market timing support
- Entry/exit opportunity identification
- Performance benchmarking

---

## References and Resources

### Recommended Reading
- Box & Jenkins (1976): Time Series Analysis: Forecasting and Control
- Hamilton (1994): Time Series Analysis
- Brockwell & Davis (2016): Introduction to Time Series and Forecasting

### Online Resources
- Statsmodels Documentation: https://www.statsmodels.org/
- yfinance GitHub: https://github.com/ranaroussi/yfinance
- ARIMA Model Reference: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

### Tools and Libraries
- Statsmodels: Time series modeling
- yfinance: Financial data retrieval
- Pandas: Data manipulation
- Matplotlib: Visualization
- SciPy: Statistical testing

---

## Troubleshooting

### Common Issues and Solutions

Issue: ModuleNotFoundError: No module named 'yfinance'
```bash
Solution: pip install yfinance
```

Issue: Data not available for date range
```python
Solution: Verify internet connection and adjust date range
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
```

Issue: Convergence warning in model fitting
```python
Solution: This is often non-critical; proceed with analysis
# Model still produces valid results
```

---

## Contact and Support

For questions or issues:
1. Review the Questions.md file
2. Check Statsmodels documentation
3. Verify data availability via yfinance

---

## License
This project is available for educational and research purposes.

---

Last Updated: February 2026
Project Status: Complete and Production-Ready
