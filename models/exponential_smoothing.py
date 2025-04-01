# models/exponential_smoothing.py

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ExponentialSmoothingModel:
    def __init__(
        self, trend: str = "add", seasonal: str = "add", seasonal_periods: int | None = None
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, df: pd.DataFrame):
        # For simplicity, use the first numeric column.
        col = df.select_dtypes(include=[np.number]).columns[0]
        # Fill initial missing values using linear interpolation for model fitting.
        series = df[col].interpolate(method="linear")
        # Determine seasonal_periods if not provided
        if (
            self.seasonal_periods is None
            and isinstance(df.index, pd.DatetimeIndex)
            and pd.infer_freq(df.index) is not None
        ):
            # For example, if frequency is 'D' assume weekly seasonality.
            freq = pd.infer_freq(df.index)
            self.seasonal_periods = 7 if freq and "D" in freq.upper() else None
        self.model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit()

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        # For each missing value, we forecast it using the fitted model.
        imputed_df = df.copy()
        col = df.select_dtypes(include=[np.number]).columns[0]
        # Identify missing indices.
        missing_idx = imputed_df[imputed_df[col].isnull()].index
        if self.model is None:
            self.fit(df)
        for idx in missing_idx:
            # For simplicity, forecast one step ahead from the previous non-missing value.
            # In a more robust solution, you might forecast over contiguous missing segments.
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() before impute().")
            forecast = self.model.forecast(steps=1)
            imputed_df.loc[idx, col] = forecast.values[0]
        return imputed_df

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
