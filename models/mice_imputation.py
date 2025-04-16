# models/mice_imputation.py

import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MICEImputationModel:
    def __init__(self, random_state: int = 0, max_iter: int = 10):
        self.imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)

    def fit(self, df: pd.DataFrame):
        self.imputer.fit(df)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_array = self.imputer.transform(df)
        imputed_df = pd.DataFrame(imputed_array, index=df.index, columns=df.columns)
        return imputed_df

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(np.mean((holdout[col] - imputed_holdout[col]) ** 2))
        mae = np.mean(np.abs(holdout[col] - imputed_holdout[col]))
        return {"rmse": rmse, "mae": mae}
