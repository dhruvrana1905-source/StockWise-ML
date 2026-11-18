import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducible synthetic stock-price dataset
np.random.seed(42)

days = 900
dates = pd.date_range("2022-01-01", periods=days, freq="B")
returns = np.random.normal(0.0005, 0.015, days)
close = 100 * np.cumprod(1 + returns)

df = pd.DataFrame({
    "date": dates,
    "close": close,
    "volume": np.random.randint(100000, 1000000, days)
})

# Feature engineering
df["return"] = df["close"].pct_change()
df["sma5"] = df["close"].rolling(5).mean()
df["sma20"] = df["close"].rolling(20).mean()
df["volatility20"] = df["return"].rolling(20).std()

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# Predict the next trading day's closing price
df["target"] = df["close"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = ["close", "volume", "return", "sma5", "sma20",
            "volatility20", "rsi14"]

X = df[features]
y = df["target"]

# Chronological train-test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    )
}

# MLflow experiment tracking
mlflow.set_experiment("StockWise_Experiments")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_name", name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))
        mlflow.log_param("features", len(features))

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print(f"\n{name}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

print("\nAll three experiments recorded in MLflow!")
