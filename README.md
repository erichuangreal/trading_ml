# trading_ml
This trading ml model aims to predict price movement and forward return in the next 5 days using OHLCV, fundamentals, and news sentiment. Trained on NASDAQ yFinance data.

Version 1
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

Output example:
Predicted 5-day return: +2.8%
Probability of positive return: 71%
Expected range: -1.0% to +6.2%