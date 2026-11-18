# StockWise-ML — Next-Day Stock Price Prediction
## Academic demo package
This project demonstrates a machine-learning pipeline that predicts the next trading day's closing price using historical OHLCV values and technical indicators.

### Dataset
`data/demo_stock_data.csv` is **synthetic demo data**, created to make the notebook runnable offline. Do not describe it as real exchange data. Replace it with a downloaded ticker dataset before making real-world performance claims.

### Models
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

### Features
- Daily return
- SMA (5 and 20 trading days)
- RSI (14)
- Rolling volatility (20)
- OHLCV

### Run
```bash
python -m pip install -r requirements.txt
jupyter notebook StockWise_ML.ipynb
```
The notebook has an optional yfinance snippet for replacing the demo dataset.

### Limitations
This is educational only, not investment advice. The demo scores are illustrative and based on synthetic data. Market prices are noisy and affected by events that historical features cannot anticipate.
