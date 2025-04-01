# models/regression_imputation.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionImputationModel:
    def __init__(self, stochastic: bool = False, noise_scale: float = 0.1):
        """
        stochastic: If True, add residual noise to predictions.
        noise_scale: Scale factor for the added noise.
        """
        self.stochastic = stochastic
        self.noise_scale = noise_scale
        self.model = None

    def fit(self, df: pd.DataFrame):
        # For simplicity, use the time index as the predictor for the first numeric column.
        col = df.select_dtypes(include=[np.number]).columns[0]
        series = df[col].copy()
        # Get indices as numbers (e.g., 0, 1, 2, …)
        X = np.arange(len(series)).reshape(-1, 1)
        # Fill missing values for training using forward fill (this is a workaround)
        y = series.ffill().bfill()
        self.model = LinearRegression().fit(X, y)
        # Store residual standard deviation if stochastic imputation is used.
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.resid_std = np.std(residuals)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy()
        col = df.select_dtypes(include=[np.number]).columns[0]
        X_all = np.arange(len(imputed_df)).reshape(-1, 1)
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() before impute().")
        predictions = self.model.predict(X_all)
        if self.stochastic:
            noise = np.random.normal(
                0, self.resid_std * self.noise_scale, size=len(imputed_df)
            )
            predictions = predictions + noise
        # Replace only missing values.
        imputed_df[col] = imputed_df[col].fillna(
            pd.Series(predictions, index=imputed_df.index)
        )
        return imputed_df

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
