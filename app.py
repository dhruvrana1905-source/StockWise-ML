import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import make_features

st.set_page_config(page_title="StockWise-ML", page_icon="📈", layout="wide")
st.title("📈 StockWise-ML")
st.subheader("Next-Day Stock Closing Price Prediction")
st.caption("Educational machine-learning demonstration — not investment advice.")

@st.cache_data
def load_data():
    return pd.read_csv("data/demo_stock_data.csv", parse_dates=["Date"])

raw = load_data()
st.info("This hosted demo uses **synthetic stock-like data**, not live NSE/BSE prices. Results are illustrative.")
with st.sidebar:
    st.header("Project settings")
    model_choice = st.selectbox("Prediction model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    test_fraction = st.slider("Test data fraction", 0.15, 0.35, 0.20, 0.05)

st.markdown("### Historical data")
c1, c2, c3 = st.columns(3)
c1.metric("Trading days", f"{len(raw):,}")
c2.metric("Latest demo close", f"₹{raw.Close.iloc[-1]:,.2f}")
c3.metric("Data type", "Synthetic demo")
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
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted[name] = model
    results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R²": r2_score(y_test, pred)})

st.markdown("### Model comparison")
metrics = pd.DataFrame(results).set_index("Model")
st.dataframe(metrics.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True)

model = fitted[model_choice]
latest_features = raw.tail(20).copy()
engineered = make_features(raw)
latest_row = engineered[feature_cols].iloc[[-1]]
prediction = float(model.predict(latest_row)[0])
last_close = float(raw["Close"].iloc[-1])
change = prediction - last_close
a, b, c = st.columns(3)
a.metric("Latest closing price", f"₹{last_close:,.2f}")
b.metric("Predicted next close", f"₹{prediction:,.2f}")
c.metric("Predicted change", f"₹{change:,.2f}", f"{change/last_close*100:.2f}%")
st.caption(f"Selected model: {model_choice}. The model is trained and tested chronologically on the included synthetic dataset.")

st.markdown("### Actual vs predicted on test data")
test_predictions = model.predict(X_test)
chart = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": test_predictions}, index=data["Date"].iloc[split:])
st.line_chart(chart, height=300)
with st.expander("View sample data"):
    st.dataframe(raw.tail(15), use_container_width=True)
st.warning("This is a classroom demonstration using synthetic data. It does not provide reliable financial forecasts or investment recommendations.")
