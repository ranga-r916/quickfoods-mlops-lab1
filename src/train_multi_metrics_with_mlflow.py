import os
import time
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "delivery_time_model.pkl")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> dict:
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    start_time = time.time()
    preds = model.predict(X_test)
    latency = (time.time() - start_time) * 1000  # ms

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "latency": latency,
        "test_size": len(X_test),
    }


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def get_model_size():
    return os.path.getsize(MODEL_PATH) / 1024  # KB


def main():
    print("=== Lab 3: Advanced MLflow Metrics ===")

    mlflow.set_experiment("quickfoods-delivery-time")

    with mlflow.start_run():
        df = load_data(DATA_PATH)
        result = train_model(df)

        # Parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)

        # Metrics
        mlflow.log_metric("mae", result["mae"])
        mlflow.log_metric("mse", result["mse"])
        mlflow.log_metric("rmse", result["rmse"])
        mlflow.log_metric("r2", result["r2"])
        mlflow.log_metric("latency_ms", result["latency"])

        # Save model
        save_model(result["model"])

        # Model size
        model_size = get_model_size()
        mlflow.log_metric("model_size_kb", model_size)

        # Log artifacts
        mlflow.log_artifact(MODEL_PATH)

        # MLflow model
        mlflow.sklearn.log_model(result["model"], "model")

        print(f"MAE: {result['mae']:.2f}")
        print(f"MSE: {result['mse']:.2f}")
        print(f"RMSE: {result['rmse']:.2f}")
        print(f"R2: {result['r2']:.2f}")
        print(f"Latency: {result['latency']:.2f} ms")
        print(f"Model size: {model_size:.2f} KB")


if __name__ == "__main__":
    main()