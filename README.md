# trading_ml
This trading ml model aims to predict price movement and forward return in the next 5 days using OHLCV, fundamentals, and news sentiment. Trained on NASDAQ yFinance data.

### Version 1
- Successfully extract and process OHLCV/fundamentals data from 20 liquid stocks on the NASDAQ
- Test baseline and train random forest model

Baseline:
1. Buy and Hold
2. Historic average (bearish X% last year, predict +X% this year)
3. Linear regression
4. Ridge

Models:
1. RandomForest
2. XGBoost

Results:
1. Random Forest
Run: random_forest_tuned_2026-08-23_19-11-38
Model: RandomForestRegressor (Tuned)
Best Params: {'randomforestregressor__max_depth': 5, 'randomforestregressor__n_estimators': 200}
Best CV Score: 0.5257392147358216
Tuning Time: 18.54s
TESTING RESULTS:
Always Up Accuracy: 0.5579
Directional Accuracy on Test Set: 0.5842
R2 Score on Test Set: 0.0300

2. XGBOOST
Run: xgboost_tuned_2026-08-23_19-11-40
Model: XGBRegressor (Tuned)
Best Params: {'xgbregressor__learning_rate': 0.01, 'xgbregressor__max_depth': 5, 'xgbregressor__n_estimators': 200}
Best CV Score: 0.5238972370334464
Tuning Time: 1.87s
TESTING RESULTS:
Always Up Accuracy: 0.5579
Directional Accuracy on Test Set: 0.5842
R2 Score on Test Set: 0.0648

### Version 2
- Problems
- Current technical indicators (EMAs, returns, and volatility) fail to consider stocks that have become overbought or near resistance/upper liquidity
- Current dataset is too small (looking to lower timeframe or add more stocks)
- Model is unable to predict 1 year of testing data because of significant market changes throughout
- Survivorship-bias: Tickers on the sp500 all experience heavy growth and market success in the past 10+ years

- Improvements
1. Increased the number of technical indicators to 20 + looking to add fundamentals next
2. Increase the number of tickers to 100 on NASDAQ
3. At the end of the year, we're testing data using a 1-year stale model. I'll focus on implementing a continuous training process in walkforward.py to ensure the model is always up-to-date.
4. Decreased training dataset to 2020-2025

### New testing method: median and rank N testing
Reads the pooled predictions a walk-forward run saved and asks a different
question from directional accuracy: on each day, did the model correctly sort
the tickers against each other?

Two tests:
  1. Median split -- for every ticker, did the model put it on the right side of
     that day's median return. Baseline is fixed at ~50% by construction, so it
     does not drift with the market the way always-up does.
  2. Top N -- take the model's highest-ranked tickers each day and see where they
     actually landed. Edited to rank by predicted return per unit of risk (pred / volatility_20d)
     to prevent the model from choosing high volatile stocks like SMCI or TTD.
     The influence of these tickers can be seen in the top N rankings, as most of these
     stocks are present in both the top and bottom N picks. UPDATED now.

### V2 conclusions
- **Directional accuracy** is not a good indicator of accurate model predictions
- The **risk adjustment factor** (ranking using future return / volatility) improved. Ranking by raw prediction picked stocks that were 2.1× more volatile than average. (more profit at less risk)
- **Walk forward** ensure the model would also be retrained as new data was fed in to ensure that the historical data is only 1 day behind. Before, if the testing data was a year long, near the end of the testing phase, the model would be using ~300 day old data.

### 1. Directional accuracy

| Setup | Model | Always-up | Edge |
|---|---|---|---|
| Jan 2025 only (19 days) | 0.5842 | 0.5579 | +2.63pp |
| Full year walk-forward (249 days) | 0.5174 | 0.5162 | +0.11pp |

### 2. Risk adjustment (XGBoost, 22 features)

| Metric | Raw pred | Risk-adjusted |
|---|---|---|
| Volatility of picks | 2.06x | 1.41x |
| Long-short spread | +15.23 bps/day | +18.81 bps/day |
| Top-3 percentile | 0.5223 | 0.5177 |

### 3. Feature expansion (9 -> 22)

| Metric | 9 features | 22 features |
|---|---|---|
| XGBoost ranking edge | +0.77pp | +1.32pp |
| XGBoost spread | +7.59 bps | +15.23 bps |
| RandomForest ranking edge | +0.80pp | +0.88pp |
| RandomForest spread | -1.79 bps | -13.69 bps |

MODEL CONCLUSION: Ridge and RandomForest models will not be tested from now on as research proves that **XGBoost** is superior.

