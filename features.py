import numpy as np
import pandas as pd

def make_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["Volatility_20"] = df["Return"].rolling(20).std()
    df["Target"] = df["Close"].shift(-1)
    return df.replace([np.inf, -np.inf], np.nan).dropna()
