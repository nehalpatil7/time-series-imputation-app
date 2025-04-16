# models/arima_imputation.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress SARIMAX warnings for convergence, etc. during imputation
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
        target_column: str | None = None,
        use_forward_backward: bool = True,
    ):
        """
        Initialize the ARIMA/SARIMA imputation model.

        Args:
            order (tuple): The (p, d, q) order of the ARIMA model.
            seasonal_order (tuple): The (P, D, Q, S) seasonal order for SARIMA.
                                    If S=0, a standard ARIMA model is used.
            target_column (str | None): The name of the column to impute.
                                        If None, the first numeric column is used.
            use_forward_backward (bool): If True, perform both forward and backward
                                         imputation and average the results.
                                         If False, only perform forward imputation.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self._target_column = target_column  # Internal storage
        self.use_forward_backward = use_forward_backward
        self.forward_model_fit = None
        self.backward_model_fit = None
        self.col_to_impute = None  # Determined during fit/transform

    def _get_target_column(self, df: pd.DataFrame) -> str:
        """Helper to determine the column to impute."""
        if self._target_column:
            if self._target_column not in df.columns:
                raise ValueError(
                    f"Target column '{self._target_column}' not found in DataFrame."
                )
            if not pd.api.types.is_numeric_dtype(df[self._target_column]):
                raise ValueError(
                    f"Target column '{self._target_column}' must be numeric."
                )
            return self._target_column
        else:
            # Default to first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.any():
                raise ValueError(
                    "No numeric columns found in DataFrame for imputation."
                )
            return numeric_cols[0]

    def _fit_model(self, series: pd.Series, order: tuple, seasonal_order: tuple):
        """Fits the ARIMA/SARIMA model, handling potential errors."""
        model = None
        model_fit = None
        # Use linear interpolation just for fitting the model if NaNs are present
        series_for_fit = series.interpolate(method="linear").ffill().bfill()
        if series_for_fit.isnull().any():
            print(
                f"Warning: Could not fill all NaNs in column {series.name} for model fitting. Remaining NaNs: {series_for_fit.isnull().sum()}"
            )
            return None  # Cannot fit if NaNs remain after interpolation

        try:
            # Check if seasonal component is active (S > 0)
            if seasonal_order[3] > 0:
                model = SARIMAX(
                    series_for_fit,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(
                    series_for_fit,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            model_fit = model.fit(disp=False)  # disp=False reduces console output
        except np.linalg.LinAlgError as e:
            print(
                f"Warning: Linear Algebra Error during ARIMA/SARIMAX fitting for column {series.name}. Skipping model. Error: {e}"
            )
        except ValueError as e:
            print(
                f"Warning: Value Error during ARIMA/SARIMAX fitting (often related to order/data) for column {series.name}. Skipping model. Error: {e}"
            )
        except Exception as e:  # Catch other potential fitting errors
            print(
                f"Warning: An unexpected error occurred during ARIMA/SARIMAX fitting for column {series.name}. Skipping model. Error: {e}"
            )

        return model_fit

    def fit(self, df: pd.DataFrame):
        """
        Determines the target column and optionally pre-fits models if needed
        (though fitting often happens dynamically in transform).

        Args:
            df (pd.DataFrame): The training DataFrame. It can contain missing values.
                               This data is primarily used to determine the target
                               column and potentially pre-fit models if desired,
                               although fitting within transform is generally safer
                               to handle different data lengths/patterns.
        """
        self.col_to_impute = self._get_target_column(df)
        print(f"ARIMA Imputation target column set to: {self.col_to_impute}")

        # Optional: Pre-fit models here if you expect transform to be called
        # multiple times on similar data structures without refitting.
        # For robustness with varying data in transform, fitting inside transform
        # might be preferred. Let's stick to fitting within transform for now.
        self.forward_model_fit = None
        self.backward_model_fit = None
        return self

    def _impute_series(self, series: pd.Series, model_fit) -> pd.Series:
        """Helper function to impute NaNs in a series using a fitted model."""
        imputed_series = series.copy()
        if model_fit is None:
            print(
                f"Warning: No valid model available for column {series.name}. Returning series with NaNs potentially unfilled."
            )
            return imputed_series  # Return original if no model

        missing_indices = imputed_series[imputed_series.isnull()].index
        if missing_indices.empty:
            return imputed_series  # No imputation needed

        # Use predict for in-sample forecasting/interpolation
        # Predict requires start and end points relative to the *original* series used for fitting
        try:
            predictions = model_fit.predict(start=series.index[0], end=series.index[-1])
            # Fill NaNs using the predictions, aligning by index
            imputed_series.fillna(predictions, inplace=True)
        except Exception as e:
            print(
                f"Warning: Error during prediction/fillna for column {series.name}. Returning series with potentially unfilled NaNs. Error: {e}"
            )

        return imputed_series

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in the specified target column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame with missing values to impute.

        Returns:
            pd.DataFrame: DataFrame with missing values in the target column imputed.
        """
        if self.col_to_impute is None:
            # Determine column if fit wasn't called or needs confirmation
            self.col_to_impute = self._get_target_column(df)
            print(
                f"ARIMA Imputation target column determined during transform: {self.col_to_impute}"
            )

        df_imputed = df.copy()
        series_to_impute = df_imputed[self.col_to_impute]

        if not series_to_impute.isnull().any():
            return df_imputed  # No missing values in target column

        # --- Forward Pass ---
        print("Fitting forward ARIMA model...")
        self.forward_model_fit = self._fit_model(
            series_to_impute, self.order, self.seasonal_order
        )
        imputed_forward = self._impute_series(series_to_impute , self.forward_model_fit)

        if not self.use_forward_backward:
            df_imputed[self.col_to_impute] = imputed_forward
            # Final check: fill any remaining NaNs (e.g., if prediction failed)
            df_imputed[self.col_to_impute] = (
                df_imputed[self.col_to_impute].ffill().bfill()
            )
            return df_imputed

        # --- Backward Pass ---
        print("Fitting backward ARIMA model...")
        series_reversed = series_to_impute.iloc[::-1]
        # Reset index for backward model fitting if it's a default RangeIndex
        # If it's a DatetimeIndex, reversing is usually sufficient
        original_index = series_reversed.index
        if not isinstance(original_index, pd.DatetimeIndex):
            series_reversed = series_reversed.reset_index(drop=True)

        # Note: Differencing order 'd' might need adjustment for reversed series
        # For simplicity, we use the same order. Advanced use might require tuning.
        self.backward_model_fit = self._fit_model(
            series_reversed, self.order, self.seasonal_order
        )
        imputed_backward_reversed = self._impute_series(
            series_reversed, self.backward_model_fit
        )

        # Reverse the backward imputation results back to original order
        imputed_backward = imputed_backward_reversed.iloc[::-1]
        imputed_backward.index = original_index  # Restore original index

        # --- Averaging ---
        print("Averaging forward and backward imputations...")
        # Average only where the original data was missing
        missing_mask = series_to_impute.isnull()
        averaged_values = (
            imputed_forward[missing_mask] + imputed_backward[missing_mask]
        ) / 2
        df_imputed.loc[missing_mask, self.col_to_impute] = averaged_values

        # Final check: fill any remaining NaNs (e.g., if both models failed or averaging resulted in NaN)
        df_imputed[self.col_to_impute] = df_imputed[self.col_to_impute].ffill().bfill()

        return df_imputed
