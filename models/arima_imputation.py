import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import io
import re
import copy
import base64
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class ARIMAImputationModel:
    """
    Imputes missing values in a time series using ARIMA or SARIMA models.
    Supports optional forward-backward imputation for potentially improved accuracy
    on internal gaps.
    """

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),  # (P, D, Q, S) - Set S>0 for seasonality
        y_column: str | None = None,
        use_forward_backward: bool = True,
        seasonality: bool = False,
    ):
        """
        Initialize the ARIMA/SARIMA imputation model.

        Args:
            order (tuple): The (p, d, q) order of the ARIMA model.
            seasonal_order (tuple): The (P, D, Q, S) seasonal order for SARIMA.
                                    If S=0, a standard ARIMA model is used.
            y_column (str | None): The name of the column to impute.
                                        If None, the first numeric column is used.
            use_forward_backward (bool): If True, perform both forward and backward
                                         imputation and average the results.
                                         If False, only perform forward imputation.
            freq (str | None): The frequency of the time series data. If None, it will
                               be inferred from the data if possible.
            trend (str | None): The trend parameter for ARIMA. Options are {'n', 'c', 't', 'ct'}
                                or None. 'c' includes constant, 't' includes linear trend.
            seasonality (bool): If True, perform seasonality analysis.
        """
        self.seasonality = seasonality
        self.order = order
        self.seasonal_order = seasonal_order
        self._target_column = y_column
        self.use_forward_backward = use_forward_backward
        self.transform_method = "boxcox"
        self.boxcox_lambda = None
        self.boxcox_offset = 0
        self.model_forward = None
        self.model_backward = None

    def extract_number(self, val):
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group(0)) if match else None

    # ------------ Diagnostic Plot Functions ------------
    def generate_diagnostic_plots(self, df: pd.DataFrame, column=None):
        """
        Generate diagnostic plots for ARIMA time series analysis:
        1. Boxcox transformation plot
        2. ACF (Autocorrelation Function) plot
        3. PACF (Partial Autocorrelation Function) plot

        Args:
            df: DataFrame containing the time series data
            column: Column name to analyze. If None, the first numeric column will be used.

        Returns:
            dict: Dictionary containing base64 encoded plot images
        """
        # Get the target column
        if column is None:
            column = self._target_column

        # Get series without NaN values for BoxCox transformation
        series = df[column].copy()
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No non-null values found in the data"}

        # Make positive for BoxCox
        min_val = clean_series.min()
        if min_val <= 0:
            # Add offset to make all values positive
            offset = abs(min_val) + 1
            clean_series = clean_series + offset
        else:
            offset = 0

        # Prepare plots
        result = {}

        # 1. BoxCox transformation plot
        try:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Original data
            axes[0].plot(clean_series.index, clean_series.values)
            axes[0].set_title("Original Data")

            # BoxCox transformation
            transformed_data, lambda_val = boxcox(clean_series.values) # type: ignore
            axes[1].plot(clean_series.index, transformed_data)
            axes[1].set_title(f"BoxCox Transformed Data (λ={lambda_val:.4f})")

            if offset > 0:
                fig.suptitle(f"BoxCox Transformation (offset applied: +{offset})")
            else:
                fig.suptitle("BoxCox Transformation")

            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            result["boxcox_plot"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            result["boxcox_error"] = str(e)

        # 2. Take the difference of the BoxCox transformed data for stationarity
        try:
            diff_transformed = pd.Series(transformed_data).diff().dropna()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(diff_transformed.index, diff_transformed.values) # type: ignore
            ax.set_title("Differenced BoxCox Transformed Data")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            result["diff_plot"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            result["diff_error"] = str(e)

        # 3. ACF plot of differenced data
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(diff_transformed, ax=ax, lags=40)
            ax.set_title("Autocorrelation Function (ACF)")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            result["acf_plot"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            result["acf_error"] = str(e)

        # 4. PACF plot of differenced data
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(diff_transformed, ax=ax, method="ywm", lags=40)
            ax.set_title("Partial Autocorrelation Function (PACF)")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            result["pacf_plot"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            result["pacf_error"] = str(e)

        return result

    def transform(self, df: pd.DataFrame, column=None):
        """
        Fit the ARIMA model to the data following the provided structure.
        Args:
            df: DataFrame containing the time series data
            column: Column name to analyze. If None, the first numeric column will be used.
            test_size: Proportion of data to use for testing (default: 0.2)
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        try:
            if not "Date" in df.columns:
                raise ValueError("No date column found in the dataframe")

            data = copy.deepcopy(df)
            data = data.set_index("Date")
            if column is None:
                column = self._target_column
                data[column] = data[column].apply(self.extract_number)
                data[column] = data[column].apply(pd.to_numeric, errors='ignore')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric (Y) columns found in the dataframe")

            mask = data[column].isnull().to_numpy()
            missing_values = data.loc[mask].copy()

            null_positions = np.where(mask)[0]
            first_pos = null_positions[0]
            last_pos = null_positions[-1]

            part1 = data.iloc[:first_pos].copy()
            part2 = data.iloc[last_pos + 1 :].copy()

            # Forward pass
            if len(part1) > 1:
                try:
                    # shift values so that all values > 0
                    min_val = part1[column].min()
                    if min_val <= 0: shift_fwd = abs(min_val) + 1e-6
                    else: shift_fwd = 0.0
                    part1[column] = part1[column] + shift_fwd

                    part1["Boxcox"], lam = boxcox(part1[column]) # type: ignore
                    part1.dropna(inplace=True)

                    if self.seasonality:
                        model = SARIMAX(
                            part1["Boxcox"],
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                        ).fit()
                    else:
                        model = ARIMA(
                            part1["Boxcox"],
                            order=self.order,
                        ).fit()
                    part1_forecast = model.forecast(len(missing_values)) # type: ignore
                    part1_imputed_value = inv_boxcox(part1_forecast, lam) - shift_fwd
                    part1_imputed_value = part1_imputed_value.tolist()

                    if len(part2) < 1:
                        missing_values.loc[mask, column] = part1_imputed_value[ # type: ignore
                            ::-1
                        ].tolist()
                        return missing_values

                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    return {"error": str(e)}

            # Backward pass (reverse the data)
            if len(part2) > 1:
                try:
                    part2 = part2.iloc[::-1].copy()

                    # Compute shift
                    min_val_b = part2[column].min()
                    if min_val_b <= 0:
                        shift_bwd = abs(min_val_b) + 1e-6
                    else:
                        shift_bwd = 0.0
                    part2[column] = part2[column] + shift_bwd

                    part2["Boxcox"], lam = boxcox(part2[column]) # type: ignore
                    part2.dropna(inplace=True)

                    if self.seasonality:
                        model = SARIMAX(
                            part2["Boxcox"],
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                        ).fit()
                    else:
                        model = ARIMA(
                            part2["Boxcox"],
                            order=self.order,
                        ).fit()
                    part2_forecast = model.forecast(len(missing_values)) # type: ignore
                    part2_imputed_value = inv_boxcox(part2_forecast, lam) - shift_bwd

                    # Reverse the backward forecasts to match the original order
                    part2_imputed_value = part2_imputed_value[::-1].tolist()
                    if len(part1) < 1:
                        missing_values.loc[mask, column] = part2_imputed_value # type: ignore
                        return missing_values

                except Exception as e:
                    print(f"Error in backward pass: {str(e)}")
                    return {"error": str(e)}

            # Average the forward & backward forecasts
            imputed_values = np.round(
                (np.array(part1_imputed_value) + np.array(part2_imputed_value)) / 2, 2
            )
            missing_values[column] = imputed_values.tolist()
            missing_values.reset_index(inplace=True)
            return missing_values

        except Exception as e:
            print(f"Error in merging forward and backward passes: {str(e)}")
            return {"error": str(e)}

    def fit(self, df: pd.DataFrame, column=None):
        """
        Fit method for API compatibility. This method calls the transform method.
        Args:
            df: DataFrame containing the time series data
            column: Column name to analyze. If None, the first numeric column will be used.
        Returns:
            self: Returns self for method chaining
        """
        # Store the column name for later use
        if column is None:
            column = self._target_column
            if column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in the dataframe")
                column = numeric_cols[0]

        self._target_column = column

        # Just store the dataframe for later use in transform
        self._df = df.copy()

        return self
