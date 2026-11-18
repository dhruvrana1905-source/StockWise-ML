
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(
    page_title="StockWise-ML",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockWise-ML")
st.subheader("NSE Next-Day Stock Price Prediction")

st.sidebar.header("Stock Settings")

ticker = st.sidebar.selectbox(
    "Select NSE Stock",
    [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "TMCV.NS",
        "BHARTIARTL.NS",
        "LT.NS"
    ]
)

period = st.sidebar.selectbox(
    "Historical Data Period",
    ["1y", "2y", "5y", "10y"],
    index=2
)

test_fraction = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20
)


@st.cache_data(ttl=3600)
def load_data(symbol, selected_period):
    df = yf.download(
        symbol,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            "No historical data was returned for this stock."
        )

    # Handle yfinance MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing market data columns: {missing}"
        )

    df = df[required_columns].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if len(df) < 100:
        raise ValueError(
            "Not enough historical data. Try another stock or period."
        )

    return df


def make_features(data):
    df = data.copy()

    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df.loc[(loss == 0) & (gain > 0), "RSI_14"] = 100
    df.loc[(loss == 0) & (gain == 0), "RSI_14"] = 50

    df["Volatility_20"] = df["Return"].rolling(20).std()
    df["Target"] = df["Close"].shift(-1)

    return df.replace([np.inf, -np.inf], np.nan)


# Download market data
try:
    with st.spinner(f"Downloading {ticker} historical NSE data..."):
        data = load_data(ticker, period)

except Exception as error:
    st.error(f"Could not download stock data: {error}")
    st.info(
        "Please try again later or select another NSE stock. "
        "Yahoo Finance data availability is not guaranteed."
    )
    st.stop()


st.success(
    f"Successfully loaded historical data for {ticker}"
)

st.caption(
    f"Data period: {data.index.min().date()} to "
    f"{data.index.max().date()} | "
    f"Trading sessions: {len(data)}"
)

feature_data = make_features(data)
feature_data = feature_data.dropna(
    subset=["SMA_5", "SMA_20", "RSI_14", "Volatility_20"]
)

st.header("Technical Indicators")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

latest = feature_data.iloc[-1]

col1.metric("SMA 5", f"₹{latest['SMA_5']:.2f}")
col2.metric("SMA 20", f"₹{latest['SMA_20']:.2f}")
col3.metric("RSI 14", f"{latest['RSI_14']:.2f}")
col4.metric("Volatility 20", f"{latest['Volatility_20']:.4f}")

feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "SMA_5", "SMA_20",
    "RSI_14", "Volatility_20"
]

# Latest available row for next-day prediction
latest_row = feature_data.dropna(
    subset=feature_columns
).iloc[[-1]]

latest_features = latest_row[feature_columns]

# Training data with known next-day closing prices
model_data = feature_data.dropna(
    subset=feature_columns + ["Target"]
).copy()

if len(model_data) < 100:
    st.error(
        "Insufficient usable data after feature engineering. "
        "Please select a longer historical period."
    )
    st.stop()

X = model_data[feature_columns]
y = model_data["Target"]
current_close = model_data["Close"]

split_index = int(
    len(model_data) * (1 - test_fraction / 100)
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
current_close_test = current_close.iloc[split_index:]

if len(X_test) < 2:
    st.error("Not enough test observations.")
    st.stop()


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = {}
predictions = {}
trained_models = {}

# Train and evaluate all models
for name, model in models.items():
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    predictions[name] = predicted
    trained_models[name] = model

    actual_direction = np.where(
        y_test.to_numpy() > current_close_test.to_numpy(),
        "Up",
        "Down"
    )

    predicted_direction = np.where(
        predicted > current_close_test.to_numpy(),
        "Up",
        "Down"
    )

    results[name] = {
        "Accuracy": accuracy_score(
            actual_direction,
            predicted_direction
        ),

        "Macro Precision": precision_score(
            actual_direction,
            predicted_direction,
            labels=["Down", "Up"],
            average="macro",
            zero_division=0
        ),

        "Macro Recall": recall_score(
            actual_direction,
            predicted_direction,
            labels=["Down", "Up"],
            average="macro",
            zero_division=0
        ),

        "Macro F1-Score": f1_score(
            actual_direction,
            predicted_direction,
            labels=["Down", "Up"],
            average="macro",
            zero_division=0
        ),

        "MAE": mean_absolute_error(
            y_test, predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(y_test, predicted)
        ),

        "R2 Score": r2_score(
            y_test, predicted
        )
    }


# Model selection
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)

results_df = pd.DataFrame(results).T


st.header("1. Model Performance Comparison")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2%}",
        "Macro Precision": "{:.2%}",
        "Macro Recall": "{:.2%}",
        "Macro F1-Score": "{:.2%}",
        "MAE": "₹{:.2f}",
        "RMSE": "₹{:.2f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True
)


st.header("2. Classification Metrics")

classification_columns = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-Score"
]

classification_df = results_df[
    classification_columns
] * 100

for metric in classification_columns:
    st.subheader(metric)
    st.bar_chart(classification_df[[metric]])


st.header("3. Regression Metrics")

for metric in ["MAE", "RMSE", "R2 Score"]:
    st.subheader(metric)
    st.bar_chart(results_df[[metric]])


st.header("4. Best Model Results")

best_directional = results_df[
    "Macro F1-Score"
].idxmax()

best_price = results_df[
    "MAE"
].idxmin()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best Directional Model",
        best_directional
    )
    st.caption("Based on highest Macro F1-Score.")

with col2:
    st.metric(
        "Best Price Prediction Model",
        best_price
    )
    st.caption("Based on lowest MAE.")


st.header("5. Next-Day Closing Price Prediction")

selected_prediction = trained_models[
    selected_model
].predict(latest_features)[0]

latest_close = float(latest_row["Close"].iloc[0])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Closing Price",
        f"₹{latest_close:.2f}"
    )

with col2:
    st.metric(
        "Predicted Next Closing Price",
        f"₹{selected_prediction:.2f}"
    )

if selected_prediction > latest_close:
    st.success("Predicted Direction: UP")
else:
    st.info("Predicted Direction: DOWN")


st.header("6. Actual vs Predicted Prices")

comparison_df = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions[selected_model]
}, index=y_test.index)

st.line_chart(comparison_df)


st.header("7. Historical NSE Data")

st.dataframe(
    feature_data.dropna(subset=feature_columns).tail(10)[feature_columns],
    use_container_width=True
)

st.caption(
    "Data source: Yahoo Finance. "
    "Prices are adjusted historical market data where available. "
    "The application is educational and not investment advice."
)
