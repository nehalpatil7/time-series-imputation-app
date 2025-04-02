# models/knn_imputation.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error


class KNNImputationModel:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.numeric_columns = None

    def fit(self, df: pd.DataFrame):
        # Store numeric columns
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        # Fit imputer on numeric columns only
        self.imputer.fit(df[self.numeric_columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid modifying the original
        df_transformed = df.copy()
        # Transform only numeric columns
        df_transformed[self.numeric_columns] = self.imputer.transform(df[self.numeric_columns])
        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.transform(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}
