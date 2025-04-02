# preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
import os


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a given file path. Supports CSV and Excel files.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        df (pd.DataFrame): Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not read file with any of the encodings: {encodings}")
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

    return df


def preprocess_dataset(df: pd.DataFrame, frequency: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Clean and preprocess the DataFrame.

    Steps:
      - If a column named 'date' exists, parse it as datetime and set as index.
      - If the index is a datetime index, sort by date.
      - Remove duplicate rows.
      - Extract key features:
          * length: number of rows.
          * missing_rate: average fraction of missing values per column.
          * has_trend: binary indicator (1/0) based on linear regression on first numeric column.
          * has_seasonality: binary indicator (1/0) if datetime index has an inferred frequency.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.
        frequency (str, optional): Expected frequency of the data. If None, will be inferred.

    Returns:
        df_processed (pd.DataFrame): Cleaned DataFrame.
        features (dict): Dictionary of extracted features.
    """
    df = df.copy()

    # If a 'date' column exists, convert it to datetime and set as index
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        except Exception as e:
            print(f"Warning: Unable to convert 'date' column to datetime. {e}")

    # If the index is datetime, sort by index
    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Extract features:
    length = len(df)
    # Compute missing rate per column and average over all columns
    missing_rate = df.isnull().mean().mean() if length > 0 else 0.0
    if missing_rate == 0.0:
        # Calculate average difference between consecutive values for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            diffs = df[numeric_cols].diff().abs()
            avg_diff = diffs.mean().mean()
            # Check if there are any large differences (outliers)
            max_diff = diffs.max().max()
            has_large_diffs = int(max_diff > 3 * avg_diff)
            if has_large_diffs > 3:
                missing_rate = has_large_diffs

    # Determine trend using the first numeric column (if available)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        # Fill missing values for trend detection using forward/backward fill
        series = df[col].ffill().bfill()
        x = np.arange(len(series))
        # Compute linear regression slope
        slope, _, _, p_value, _ = linregress(x, series)
        # Mark trend as present if the absolute slope is above a small threshold and statistically significant.
        has_trend = int(abs(slope) > 1e-3 and p_value < 0.05)
    else:
        has_trend = 0

    # Determine seasonality: if index is datetime and an inferred frequency is available, assume seasonality.
    if isinstance(df.index, pd.DatetimeIndex):
        inferred_freq = pd.infer_freq(df.index)
        has_seasonality = int(inferred_freq is not None)
    else:
        has_seasonality = 0

    features = {
        "length": length,
        "missing_rate": missing_rate,
        "has_trend": has_trend,
        "has_seasonality": has_seasonality,
    }

    return df, features


# For testing the preprocessing module standalone.
if __name__ == "__main__":
    file_path = (
        "AirPassengers.csv"
    )
    try:
        df_raw = load_dataset(file_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        df_raw = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )
        print("Using generated dummy dataset.")

    df_processed, features = preprocess_dataset(df_raw, frequency=None)
    print("Preprocessing complete. Extracted features:")
    print(features)
    print("First 5 rows of processed data:")
    print(df_processed.head())
