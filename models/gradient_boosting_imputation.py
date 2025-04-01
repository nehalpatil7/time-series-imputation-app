# models/gradient_boosting_imputation.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


class GradientBoostingImputationModel:
    def __init__(self, params: dict | None = None, num_boost_round: int = 50):
        # Default parameters can be adjusted
        self.params = params or {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.1,
            "verbosity": 0,
        }
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, df: pd.DataFrame):
        # For simplicity, we impute the first numeric column using its index as feature.
        col = df.select_dtypes(include=[np.number]).columns[0]
        series = df[col].copy()
        X = np.arange(len(series)).reshape(-1, 1)
        # Use rows where value is not missing.
        mask = ~series.isnull()
        dtrain = xgb.DMatrix(X[mask], label=series[mask])
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round
        )

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy()
        col = df.select_dtypes(include=[np.number]).columns[0]
        X_all = np.arange(len(df)).reshape(-1, 1)
        dtest = xgb.DMatrix(X_all)
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() before impute().")
        predictions = self.model.predict(dtest)
        # Replace missing values only.
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
