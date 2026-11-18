# Lightweight reusable feature pipeline
import pandas as pd
def add_features(data):
    d=data.copy().sort_values("Date")
    d["Return_1d"]=d["Close"].pct_change()
    d["SMA_5"]=d["Close"].rolling(5).mean()
    d["SMA_20"]=d["Close"].rolling(20).mean()
    delta=d["Close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/(loss+1e-9)
    d["RSI_14"]=100-100/(1+rs)
    d["Volatility_20"]=d["Return_1d"].rolling(20).std()
    d["Target"]=d["Close"].shift(-1)
    return d.dropna()
