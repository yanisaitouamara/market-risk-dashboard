# Market Risk Dashboard
### VaR, Expected Shortfall, GARCH & Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This project builds a comprehensive market risk dashboard 
for a diversified portfolio of 5 international equities 
(AAPL, JPM, MC.PA, NESN.SW, TTE.PA) over the period 
2015-2024.

It combines classical risk management methods with 
modern machine learning techniques to measure, predict 
and explain market risk.

---

## Portfolio
| Ticker | Company | Sector | Geography |
|--------|---------|--------|-----------|
| AAPL | Apple | Technology | USA |
| JPM | JPMorgan | Banking | USA |
| MC.PA | LVMH | Luxury | France |
| NESN.SW | Nestlé | Consumer Staples | Switzerland |
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
Historical prices downloaded via yfinance (2015-2024).
Log-returns computed and missing values handled with forward-fill.

### Step 2 — Value at Risk (VaR) — 4 Methods
- **Parametric VaR** — assumes normal distribution and constant volatility
- **Historical VaR** — empirical quantile, no distribution assumption
- **Monte Carlo VaR** — 10,000 simulated return scenarios
- **Dynamic GARCH VaR** — time-varying volatility via GARCH(1,1)

### Step 3 — Expected Shortfall (CVaR)
Average loss beyond the VaR threshold.
ES averages **1.5x the VaR** across all assets, justifying Basel III's shift from VaR to ES as the standard regulatory risk measure.

### Step 4 — GARCH(1,1) Volatility Modeling
Dynamic volatility estimation using the arch library.

Key findings:
| Ticker | Alpha | Beta | Alpha+Beta |
|--------|-------|------|------------|
| AAPL | 0.11 | 0.84 | 0.95 |
| JPM | 0.12 | 0.81 | 0.93 |
| MC.PA | 0.05 | 0.92 | 0.97 |
| NESN.SW | 0.08 | 0.85 | 0.93 |
| TTE.PA | 0.08 | 0.91 | 0.99 |

TotalEnergies shows the highest volatility persistence (α+β = 0.99), consistent with oil price exposure.

### Step 5 — Machine Learning: Stress Day Prediction
Binary classification to predict whether tomorrow will be a stress day (return < VaR).

**Features:**
- 5-day and 20-day cumulative returns
- 5-day and 20-day realized volatility
- GARCH conditional volatility
- Lagged returns (lag 1, 2, 3)

**Results:**
| Model | AUC-ROC | Recall (stress days) |
|-------|---------|----------------------|
| XGBoost | 0.986 | 50% |
| Random Forest | 0.983 | 0% |

XGBoost clearly outperforms Random Forest — it detects stress days with high confidence.

### Step 6 — SHAP Explainability
Top predictors of stress days identified by SHAP:
1. **return_5d** — 5-day cumulative return (most important)
2. **realized_vol_5d** — short-term realized volatility
3. **realized_vol_20d** — medium-term realized volatility
6. **garch_vol** — GARCH conditional volatility

Recent negative returns combined with high short-term volatility are the strongest stress signals.

### Step 7 — Backtesting: Kupiec & Christoffersen
| Ticker | Exceptions | Rate | Kupiec | Christoffersen | Verdict |
|--------|-----------|------|--------|----------------|---------|
| AAPL | 130 | 5.04% | PASS | CLUSTERING | ⚠️ |
| JPM | 130 | 5.04% | PASS | CLUSTERING | ⚠️ |
| MC.PA | 130 | 5.04% | PASS | CLUSTERING | ⚠️ |
| NESN.SW | 130 | 5.04% | PASS | PASS | ✅ |
| TTE.PA | 130 | 5.04% | PASS | CLUSTERING | ⚠️ |

VaR is correctly calibrated in frequency (Kupiec ✅) but exceptions cluster during crisis periods for 4/5 assets (Christoffersen ⚠️), justifying the use of dynamic GARCH-based VaR.

---

## Key Findings

1. **Nestlé is the least risky asset** across all methods — lowest volatility (1.01%), lowest VaR (-1.57%), only asset passing both backtests.
2. **ES averages 1.5x VaR** — VaR alone underestimates tail risk.
3. **Static VaR is correctly calibrated in frequency** but misses volatility clustering.
4. **GARCH captures market regimes** — VaR expands during crises and contracts during calm periods.
5. **XGBoost predicts stress days with AUC-ROC = 0.986** — 5-day returns and short-term volatility are the strongest predictors.

---

## Global Conclusion

You cannot rely on a single risk measure:
- **VaR** gives the threshold
- **ES** gives the severity
- **GARCH** gives the dynamics
- **Machine Learning** gives the early warning signal

Together they form a complete market risk framework.

---

## Requirements
```
pip install yfinance pandas numpy scipy matplotlib arch scikit-learn xgboost shap
```

---

## Tools & Libraries
| Library | Usage |
|---------|-------|
| yfinance | Data collection |
| pandas, numpy | Data manipulation |
| scipy | Statistical tests |
| arch | GARCH modeling |
| scikit-learn | Machine Learning |
| xgboost | XGBoost classifier |
| shap | Explainability |
| matplotlib | Visualization |

---

## References
- Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models
- Christoffersen, P. (1998). Evaluating Interval Forecasts
- Basel III Framework — BIS (2019)

---

## Author
**Yanis Ait Ouamara**  
Master 2 Finance Technology Data  
University Paris 1 Panthéon-Sorbonne  
aitouamara.yanis@gmail.com
