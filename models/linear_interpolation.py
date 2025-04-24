# models/linear_interpolation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer


class LinearInterpolationModel:
    def __init__(self):
        pass

    def impute(self, df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        return df.interpolate(method=method, limit_direction="forward", limit_area="inside", inplace=True)

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout, method="linear")
        # Evaluate on the first numeric column.
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
