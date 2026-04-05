import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_series(csv_path: str, date_col: str, target_col: str) -> pd.DataFrame:
    """
    Load a time series CSV, parse date column, sort by date,
    and keep only date + target columns.
    """
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df[[date_col, target_col]].copy()


def train_test_split_series(df: pd.DataFrame, target_col: str, test_size: int = 24):
    """
    Split a univariate time series into train and test parts,
    keeping chronological order.
    """
    y = df[target_col].astype(float)
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]
    return y_train, y_test


def scale_series(y_train: pd.Series, y_test: pd.Series):
    """
    Fit scaler on train only, then transform train and test.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train = y_train.to_frame()
    test = y_test.to_frame()

    scaler.fit(train)

    y_train_scaled = scaler.transform(train)
    y_test_scaled = scaler.transform(test)

    return y_train_scaled, y_test_scaled, scaler


def create_dataset(series: np.ndarray, look_back: int):
    """
    Convert a 1D scaled series into supervised learning sequences.

    Example:
    look_back = 3
    [1,2,3,4,5] -> X: [[1,2,3],[2,3,4]], y: [4,5]
    """
    X, y = [], []

    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])

    return np.array(X), np.array(y)


def inverse_transform_array(arr: np.ndarray, scaler: MinMaxScaler):
    """
    Inverse-transform a 1D/2D array back to original scale.
    """
    arr = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()
