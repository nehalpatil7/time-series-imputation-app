import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def split_data(df: pd.DataFrame, holdout_percent: float = 0.15):
    """
    Splits the DataFrame into training and holdout sets based on time order.

    Parameters:
    - df: Input DataFrame (time-series data sorted by time).
    - holdout_percent: Fraction of data reserved as holdout.

    Returns:
    - train: Training DataFrame.
    - holdout: Holdout (evaluation) DataFrame.
    """
    n = len(df)
    split_index = int(n * (1 - holdout_percent))
    train = df.iloc[:split_index, :].copy()
    holdout = df.iloc[split_index:, :].copy()
    return train, holdout


def compute_metrics(true_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics on the imputed holdout data.

    Parameters:
    - true_df: The original holdout DataFrame with ground truth.
    - imputed_df: The imputed DataFrame over the same holdout period.

    Returns:
    - A dictionary containing RMSE and MAE computed on the first numeric column.
    """
    # Select the first numeric column
    col = true_df.select_dtypes(include=["number"]).columns[0]
    rmse = np.sqrt(mean_squared_error(true_df[col], imputed_df[col]))
    mae = mean_absolute_error(true_df[col], imputed_df[col])
    return {"rmse": rmse, "mae": mae}
