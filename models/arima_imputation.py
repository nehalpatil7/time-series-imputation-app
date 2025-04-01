# models/arima_imputation.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ARIMAImputationModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, df: pd.DataFrame):
        # Use the first numeric column
        col = df.select_dtypes(include=[np.number]).columns[0]
        series = df[col].interpolate(method="linear")
        # Fit ARIMA model on the complete series (after interpolation)
        self.model = ARIMA(series, order=self.order).fit()

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy()
        col = df.select_dtypes(include=[np.number]).columns[0]
        # Identify missing indices.
        missing_idx = imputed_df[imputed_df[col].isnull()].index
        if self.model is None:
            self.fit(df)
        for idx in missing_idx:
            # Use the ARIMA model to forecast one step ahead.
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() before impute()")
            forecast = self.model.forecast(steps=1)
            imputed_df.loc[idx, col] = forecast.values[0]
            # Optionally, update the model with the imputed value (online update not implemented here)
        return imputed_df

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
