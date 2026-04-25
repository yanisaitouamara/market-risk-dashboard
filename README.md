# Market Risk Dashboard
### VaR, Expected Shortfall, GARCH and Machine Learning

## Overview

This project builds a comprehensive market risk framework for a diversified portfolio of 5 international equities over the period 2015-2024. It combines classical risk management methods with modern machine learning techniques to measure, predict and explain market risk.

The project was built as part of the Master 2 Finance Technology Data application at University Paris 1 Panthéon-Sorbonne.

---

## Portfolio

| Ticker | Company | Sector | Geography |
|--------|---------|--------|-----------|
| AAPL | Apple | Technology | USA |
| JPM | JPMorgan | Banking | USA |
| MC.PA | LVMH | Luxury | France |
| NESN.SW | Nestle | Consumer Staples | Switzerland |
| TTE.PA | TotalEnergies | Energy | France |

---

## Project Structure

```
market-risk-dashboard/
│
├── notebooks/
│   └── market_risk.ipynb       # Main Jupyter Notebook
│
├── outputs/
│   ├── returns_plot.png
│   ├── var_garch_plot.png
│   ├── stress_probability.png
│   ├── shap_importance.png
│   └── shap_beeswarm.png
│
└── README.md
```

---

## Methodology

### Step 1 — Data Collection
Historical closing prices were downloaded via yfinance for the period January 2015 to December 2024. Missing values arising from differences in trading calendars between the US, France and Switzerland were handled using forward-fill. Daily log-returns were computed to ensure stationarity.

### Step 2 — Value at Risk: Four Methods

**Parametric VaR** assumes returns follow a normal distribution with constant mean and volatility. It is the simplest approach and serves as a benchmark, but assumes the same volatility in calm and crisis periods — a clearly unrealistic assumption.

**Historical VaR** makes no distributional assumption. It takes the empirical quantile of observed returns directly. More realistic, but entirely dependent on the historical sample.

**Monte Carlo VaR** simulates 10,000 return scenarios and takes the empirical quantile. With a normal distribution assumption, it converges to the parametric result. Its real power appears when simulating non-normal distributions with fat tails or jumps.

**Dynamic GARCH VaR** uses a GARCH(1,1) model to estimate time-varying volatility. This allows the VaR to adapt in real time — expanding during crises and contracting during calm periods.

### Step 3 — Expected Shortfall

The Expected Shortfall (CVaR) measures the average loss in the worst 5% of scenarios — answering the question VaR cannot: how bad are losses when the threshold is breached? Across all assets, ES averages 1.5x the VaR, confirming that VaR alone underestimates tail risk. This is why Basel III adopted ES as the standard regulatory risk measure.

### Step 4 — GARCH(1,1) Volatility Modeling

Key results from GARCH estimation:

| Ticker | Alpha | Beta | Alpha+Beta |
|--------|-------|------|------------|
| AAPL | 0.11 | 0.84 | 0.95 |
| JPM | 0.12 | 0.81 | 0.93 |
| MC.PA | 0.05 | 0.92 | 0.97 |
| NESN.SW | 0.08 | 0.85 | 0.93 |
| TTE.PA | 0.08 | 0.91 | 0.99 |

All models are stationary (alpha + beta < 1). TotalEnergies shows the highest volatility persistence (0.99), consistent with its exposure to oil price shocks. JPMorgan shows the highest sensitivity to new shocks (alpha = 0.12), reflecting banks' sensitivity to economic news.

### Step 5 — Machine Learning: Stress Day Prediction

We train a binary classifier to predict whether tomorrow will be a stress day — defined as a day where the equally-weighted portfolio return falls below its historical VaR. This is a practical early warning system for risk managers.

**Features:**
- 5-day and 20-day cumulative portfolio returns
- 5-day and 20-day realized volatility
- GARCH conditional volatility
- Lagged returns (lag 1, 2, 3)

**Results:**

| Model | AUC-ROC | Recall on stress days |
|-------|---------|----------------------|
| XGBoost | 0.986 | 50% |
| Random Forest | 0.983 | 0% |

XGBoost clearly outperforms Random Forest. Despite a high AUC-ROC for both models, Random Forest never crosses the classification threshold due to the limited number of stress days in the test set (10 observations). XGBoost detects 5 out of 10 stress days with high confidence, and its probability spikes align visually with actual stress events.

### Step 6 — SHAP Explainability

SHAP (SHapley Additive Explanations) decomposes each XGBoost prediction into the individual contribution of each feature. This is essential for regulatory compliance and model validation — in risk management, it is not enough to know what the model predicts, you need to explain why.

Top predictors of stress days:
1. return_5d — 5-day cumulative return (most important by a large margin)
2. realized_vol_5d — short-term realized volatility
3. realized_vol_20d — medium-term realized volatility

Recent negative returns combined with elevated short-term volatility are the strongest early warning signals for stress days.

### Step 7 — Backtesting: Kupiec and Christoffersen Tests

| Ticker | Exceptions | Rate | Kupiec p | Christoffersen p | Verdict |
|--------|-----------|------|----------|-----------------|---------|
| AAPL | 130 | 5.04% | 0.9317 | 0.0177 | Clustering |
| JPM | 130 | 5.04% | 0.9317 | 0.0000 | Clustering |
| MC.PA | 130 | 5.04% | 0.9317 | 0.0008 | Clustering |
| NESN.SW | 130 | 5.04% | 0.9317 | 0.0921 | Pass |
| TTE.PA | 130 | 5.04% | 0.9317 | 0.0069 | Clustering |

All assets pass the Kupiec test — the VaR is correctly calibrated in frequency, with an exception rate of 5.04% against a theoretical target of 5%. However, 4 out of 5 assets fail the Christoffersen test — exceptions cluster during crisis periods rather than occurring independently. Only Nestle passes both tests, confirming its defensive profile. This clustering is precisely what justifies the use of dynamic GARCH-based VaR over static methods.

---

## Key Findings

1. Nestle is the least risky asset across all methods — lowest volatility (1.01%), lowest VaR (-1.57%), and the only asset passing both backtesting criteria.
2. Expected Shortfall averages 1.5x the VaR — VaR alone materially underestimates tail risk.
3. Static VaR is correctly calibrated in frequency but fails to capture volatility clustering during crisis periods.
4. GARCH dynamic VaR adapts in real time to market conditions, expanding during crises and contracting during calm periods.
5. XGBoost predicts stress days with an AUC-ROC of 0.986 — 5-day returns and short-term volatility are the strongest predictors.

---

## Global Conclusion

No single risk measure is sufficient on its own. VaR gives the loss threshold, Expected Shortfall gives the severity of losses beyond that threshold, GARCH gives the real-time dynamics of risk, and Machine Learning provides an early warning signal before stress materializes. Together they form a complete and robust market risk framework that addresses both regulatory requirements and practical risk management needs.

---

## Requirements

```
pip install yfinance pandas numpy scipy matplotlib arch scikit-learn xgboost shap
```

---

## Tools and Libraries

| Library | Usage |
|---------|-------|
| yfinance | Historical price data |
| pandas, numpy | Data manipulation |
| scipy | Statistical backtesting tests |
| arch | GARCH volatility modeling |
| scikit-learn | Random Forest and train/test split |
| xgboost | XGBoost classifier |
| shap | Model explainability |
| matplotlib | Visualization |

---

## References

- Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. Journal of Derivatives.
- Christoffersen, P. (1998). Evaluating Interval Forecasts. International Economic Review.
- Basel III Framework — Bank for International Settlements (2019).

---

## Author

Yanis Ait Ouamara  
Master 2 Finance Technology Data  
University Paris 1 Panthéon-Sorbonne  
aitouamara.yanis@gmail.com
