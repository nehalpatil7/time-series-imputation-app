# models/spline_interpolation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SplineImputationModel:
    def __init__(self, order: int = 3):
        self.order = order

    def fit(self, df: pd.DataFrame):
        # Spline interpolation is non-parametric.
        pass

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            imputed_df = df.interpolate(method="spline", order=self.order)
        except Exception as e:
            imputed_df = df.interpolate(method="linear")
        return imputed_df

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
