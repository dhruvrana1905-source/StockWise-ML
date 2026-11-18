import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="StockWise-ML", page_icon="📈", layout="wide")
st.title("📈 StockWise-ML")
st.subheader("Next-Day Stock Closing Price Prediction")
st.caption("Educational machine-learning demonstration — not investment advice.")

@st.cache_data
def load_data():
    # Self-contained synthetic demo data: no external CSV file is required.
    rng = np.random.default_rng(42)
    n = 900
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = rng.normal(0.0004, 0.012, n)
    close = 150 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, n))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, n))
    volume = rng.integers(500_000, 5_000_000, n)
    return pd.DataFrame({
        "Date": dates, "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume
    })

def make_features(raw):
    df = raw.copy()
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

raw = load_data()
st.info("This hosted demo uses **synthetic stock-like data**, not live NSE/BSE prices. Results are illustrative.")
with st.sidebar:
    st.header("Project settings")
    model_choice = st.selectbox("Prediction model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    test_fraction = st.slider("Test data fraction", 0.15, 0.35, 0.20, 0.05)

st.markdown("### Historical data")
col1, col2, col3 = st.columns(3)
col1.metric("Trading days", f"{len(raw):,}")
col2.metric("Latest demo close", f"₹{raw.Close.iloc[-1]:,.2f}")
col3.metric("Data type", "Synthetic demo")
st.line_chart(raw.set_index("Date")["Close"], height=280)

data = make_features(raw)
feature_cols = ["Open", "High", "Low", "Close", "Volume", "Return", "SMA_5", "SMA_20", "RSI_14", "Volatility_20"]
split = int(len(data) * (1 - test_fraction))
X_train, X_test = data[feature_cols].iloc[:split], data[feature_cols].iloc[split:]
y_train, y_test = data["Target"].iloc[:split], data["Target"].iloc[split:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.04, random_state=42)
}
results, fitted = [], {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    fitted[name] = estimator
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R²": r2_score(y_test, predictions)
    })

st.markdown("### Model comparison")
st.dataframe(pd.DataFrame(results).set_index("Model").style.format("{:.3f}"), use_container_width=True)

selected_model = fitted[model_choice]
latest_features = data[feature_cols].iloc[[-1]]
predicted_close = float(selected_model.predict(latest_features)[0])
last_close = float(raw["Close"].iloc[-1])
change = predicted_close - last_close
a, b, c = st.columns(3)
a.metric("Latest closing price", f"₹{last_close:,.2f}")
b.metric("Predicted next close", f"₹{predicted_close:,.2f}")
c.metric("Predicted change", f"₹{change:,.2f}", f"{change / last_close * 100:.2f}%")

st.caption(f"Selected model: {model_choice}. Training and testing use a chronological split.")
st.markdown("### Actual vs predicted on test data")
test_predictions = selected_model.predict(X_test)
comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": test_predictions
}, index=data["Date"].iloc[split:])
st.line_chart(comparison, height=300)

with st.expander("View sample historical data"):
    st.dataframe(raw.tail(15), use_container_width=True)

st.warning("Classroom demonstration only. The dataset is synthetic and this app does not provide reliable financial forecasts or investment recommendations.")

