# test_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.linear_interpolation import LinearInterpolationModel
from models.regression_imputation import RegressionImputationModel
from models.knn_imputation import KNNImputationModel
from models.arima_imputation import ARIMAImputationModel
from models.gradient_boosting_imputation import GradientBoostingImputationModel
from models.custom_genai_imputation import CustomGenAIImputationModel
from preprocessing import load_dataset


def introduce_missing_values(df, col, missing_fraction=0.1, random_state=42):
    """
    Randomly sets a fraction of the values in the specified column to NaN.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name where missing values will be introduced.
        missing_fraction (float): Fraction of total rows to set as missing.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with missing values introduced (randomly).
        missing_idx (Index): The indices where values were set to NaN.
    """
    np.random.seed(random_state)
    df_missing = df.copy()
    n = len(df_missing)
    random_missing_indices = np.random.choice(
        n, size=int(n * missing_fraction), replace=False
    )
    df_missing.loc[df_missing.index[random_missing_indices], col] = np.nan
    return df_missing, df_missing.index[random_missing_indices]


def introduce_block_missing_values(df, col, block_start, block_end):
    """
    Introduce a contiguous block of missing values in the specified column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column name to modify.
        block_start (int): Starting index (integer location) of block missing.
        block_end (int): Ending index (integer location) of block missing.

    Returns:
        pd.DataFrame: DataFrame with the block set to NaN.
        block_missing_idx (Index): The indices for the block missing values.
    """
    df_block_missing = df.copy()
    block_missing_idx = df_block_missing.index[block_start:block_end]
    df_block_missing.loc[block_missing_idx, col] = np.nan
    return df_block_missing, block_missing_idx


def plot_data_side_by_side(df_original, df_missing, df_imputed, col, imputed_indices):
    """
    Plots the original, missing, and imputed data side by side.
    On the imputed plot, highlights the imputed points in a different color.

    Parameters:
        df_original (pd.DataFrame): Original DataFrame.
        df_missing (pd.DataFrame): DataFrame with missing values.
        df_imputed (pd.DataFrame): DataFrame after imputation.
        col (str): Column name to plot.
        imputed_indices (Index): Indices that were imputed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle("Data Comparison: Original, Missing, and Imputed", fontsize=16)

    # Original data plot
    axes[0].plot(
        df_original.index, df_original[col], color="blue", marker="o", markersize=3
    )
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)

    # Data with missing values plot
    axes[1].plot(
        df_missing.index, df_missing[col], color="orange", marker="o", markersize=3
    )
    axes[1].set_title("Data with Missing Values")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    # Imputed data plot
    axes[2].plot(
        df_imputed.index,
        df_imputed[col],
        color="green",
        marker="o",
        markersize=3,
        label="Imputed Data",
    )

    axes[2].set_title("Regenerated Data (Imputed)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # Step 1: Load the dataset using our improved load_dataset function
    file_path = "datasets/AirPassengers.csv"
    try:
        df = load_dataset(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # If a 'Date' column exists, convert it to datetime and set it as the index.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print("No datetime information available; proceeding with index as is.")

    # Assume the first numeric column is the stock price.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found in the dataset.")
        return
    price_col = numeric_cols[0]

    # Plot the original data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_col], color="blue", marker="o", markersize=3)
    plt.title("Original Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    # Step 2: Introduce missing values:
    # a) Random missing values (10% of rows)
    df_random_missing, random_missing_idx = introduce_missing_values(
        df, price_col, missing_fraction=0.5
    )

    # b) A block missing portion: e.g., from row 50 to row 70
    df_missing, block_missing_idx = introduce_block_missing_values(
        df_random_missing, price_col, block_start=50, block_end=70
    )

    # Combine missing indices (block and random) for later highlighting.
    imputed_indices = block_missing_idx.union(random_missing_idx)

    # Plot data with missing values (side by side, we use individual plots here for clarity)
    plt.figure(figsize=(12, 6))
    plt.plot(
        df_missing.index,
        df_missing[price_col],
        color="orange",
        marker="o",
        markersize=3,
    )
    plt.title("NFLX Data with Missing Values")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    # Step 3: Use Linear Interpolation to impute missing values.
    lin_interp_model = CustomGenAIImputationModel()
    lin_interp_model.fit(df_missing)  # Not necessary, but for consistency.
    df_imputed = lin_interp_model.impute(df_missing)

    # Plot all three side by side with imputed points highlighted
    plot_data_side_by_side(df, df_missing, df_imputed, price_col, imputed_indices)


if __name__ == "__main__":
    main()
