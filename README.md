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

### New testing method
