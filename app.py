
import streamlit as st
import pandas as pd
import numpy as np

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
st.subheader("Next-Day Stock Price Prediction and Model Comparison")

st.warning(
    "This version uses synthetic demonstration data, not real NSE market data. "
    "Do not interpret its scores as real-market performance."
)


@st.cache_data
def load_data():
    np.random.seed(42)

    days = 900
    dates = pd.bdate_range(start="2022-01-03", periods=days)

    returns = np.random.normal(0.0005, 0.015, days)
    close = 100 * np.cumprod(1 + returns)

    open_price = close * (1 + np.random.normal(0, 0.003, days))
    high = np.maximum(open_price, close) * (
        1 + np.random.uniform(0, 0.01, days)
    )
    low = np.minimum(open_price, close) * (
        1 - np.random.uniform(0, 0.01, days)
    )
    volume = np.random.randint(100000, 1000000, days)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })


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


data = load_data()
featured_data = make_features(data)

feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "SMA_5", "SMA_20",
    "RSI_14", "Volatility_20"
]

# Keep the latest row for the next-day prediction.
latest_row = featured_data.dropna(subset=feature_columns).iloc[[-1]]
latest_features = latest_row[feature_columns]

# Training and testing data must have a known next-day closing price.
model_data = featured_data.dropna(
    subset=feature_columns + ["Target"]
).copy()

X = model_data[feature_columns]
y = model_data["Target"]
current_close = model_data["Close"]

test_fraction = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20
)

split_index = int(len(model_data) * (1 - test_fraction / 100))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
current_close_test = current_close.iloc[split_index:]

if len(X_test) < 2:
    st.error("Not enough test data. Increase the test percentage.")
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

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    predictions[name] = predicted

    # Convert price predictions into Up/Down labels.
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
            actual_direction, predicted_direction
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
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R2 Score": r2_score(y_test, predicted)
    }


# Sidebar model selection
selected_model = st.sidebar.selectbox(
    "Select prediction model",
    list(models.keys())
)

# Main model comparison table
st.header("1. Model Performance Comparison")

results_df = pd.DataFrame(results).T
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

# Classification metrics
st.header("2. Classification Metrics")

classification_columns = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-Score"
]

classification_df = results_df[classification_columns] * 100

for metric in classification_columns:
    st.subheader(metric)
    st.bar_chart(classification_df[[metric]])

# Regression metrics
st.header("3. Regression Metrics")

regression_columns = ["MAE", "RMSE", "R2 Score"]

for metric in regression_columns:
    st.subheader(metric)
    st.bar_chart(results_df[[metric]])

# Best models
best_directional_model = results_df["Macro F1-Score"].idxmax()
best_price_model = results_df["MAE"].idxmin()

st.header("4. Best Model Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Best Directional Model", best_directional_model)
    st.caption("Selected using the highest Macro F1-Score.")

with col2:
    st.metric("Best Price Prediction Model", best_price_model)
    st.caption("Selected using the lowest MAE.")

# Next-day prediction
st.header("5. Next-Day Closing Price Prediction")

selected_prediction = models[selected_model].predict(latest_features)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Closing Price",
        f"₹{latest_row['Close'].iloc[0]:.2f}"
    )

with col2:
    st.metric(
        f"{selected_model} Predicted Next Close",
        f"₹{selected_prediction:.2f}"
    )

if selected_prediction > latest_row["Close"].iloc[0]:
    st.success("Predicted direction: UP")
else:
    st.info("Predicted direction: DOWN")

# Actual vs predicted chart
st.header("6. Actual vs Predicted Prices")

comparison_df = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions[selected_model]
})

st.line_chart(comparison_df)

# Dataset preview
st.header("7. Historical Data Preview")
st.dataframe(data.tail(10), use_container_width=True)

st.caption(
    "Educational project only. Synthetic data is used in this version. "
    "This application is not investment advice."
)
