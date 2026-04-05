import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import (
    load_series,
    train_test_split_series,
    scale_series,
    create_dataset,
    inverse_transform_array
)
from models import build_lstm, build_bilstm, build_gru, fit_model
from evaluate import regression_metrics


# =========================
# Reproducibility
# =========================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# Config
# =========================
DATA_PATH = "data/revenue.csv"
DATE_COL = "date"
TARGET_COL = "revenue"

LOOK_BACK = 12
TEST_SIZE = 24
EPOCHS = 100
BATCH_SIZE = 16
UNITS = 64

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_losses(histories: dict, save_path: str):
    """
    Plot train/validation losses of all models.
    """
    plt.figure(figsize=(12, 6))

    for model_name, history in histories.items():
        plt.plot(history.history["loss"], label=f"{model_name} - train")
        plt.plot(history.history["val_loss"], label=f"{model_name} - val")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(dates, actual, predictions_dict, save_path: str):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual")

    for model_name, preds in predictions_dict.items():
        plt.plot(dates, preds, label=model_name)

    plt.title("Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # 1) Load data
    df = load_series(DATA_PATH, DATE_COL, TARGET_COL)

    # 2) Train/test split
    y_train, y_test = train_test_split_series(df, TARGET_COL, test_size=TEST_SIZE)

    # 3) Scale
    y_train_scaled, y_test_scaled, scaler = scale_series(y_train, y_test)

    # 4) Build supervised datasets
    X_train, y_train_seq = create_dataset(y_train_scaled, LOOK_BACK)

    # To predict the test period correctly, we combine the last LOOK_BACK
    # points of train with all test points.
    full_test_input = np.concatenate([y_train_scaled[-LOOK_BACK:], y_test_scaled], axis=0)
    X_test, y_test_seq = create_dataset(full_test_input, LOOK_BACK)

    # 5) Shapes
    input_shape = (X_train.shape[1], X_train.shape[2])

    # 6) Models
    models = {
        "LSTM": build_lstm(input_shape=input_shape, units=UNITS),
        "BiLSTM": build_bilstm(input_shape=input_shape, units=UNITS),
        "GRU": build_gru(input_shape=input_shape, units=UNITS),
    }

    histories = {}
    metrics_rows = []
    predictions_dict = {}

    # Inverse-transform true test values
    y_test_true = inverse_transform_array(y_test_seq, scaler)

    # Dates aligned with y_test_seq
    test_dates = df[DATE_COL].iloc[-len(y_test_true):].reset_index(drop=True)

    # 7) Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        history = fit_model(
            model,
            X_train,
            y_train_seq,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2
        )
        histories[model_name] = history

        # Predict on test sequences
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = inverse_transform_array(y_pred_scaled, scaler)

        predictions_dict[model_name] = y_pred

        # Metrics
        metrics = regression_metrics(y_test_true, y_pred)
        metrics["model"] = model_name
        metrics_rows.append(metrics)

    # 8) Save metrics
    metrics_df = pd.DataFrame(metrics_rows)[["model", "RMSE", "MAE", "MAPE", "R2"]]
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # 9) Save predictions
    pred_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test_true
    })

    for model_name, preds in predictions_dict.items():
        pred_df[f"{model_name}_pred"] = preds

    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred_df.to_csv(predictions_path, index=False)

    # 10) Save plots
    plot_losses(histories, os.path.join(OUTPUT_DIR, "training_curves.png"))
    plot_predictions(
        dates=test_dates,
        actual=y_test_true,
        predictions_dict=predictions_dict,
        save_path=os.path.join(OUTPUT_DIR, "forecast_plot.png")
    )

    # 11) Print summary
    print("\nMetrics:")
    print(metrics_df)


if __name__ == "__main__":
    main()
