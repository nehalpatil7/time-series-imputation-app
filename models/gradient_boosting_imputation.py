import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


class GradientBoostingImputationModel:
    """
    Gradient Boosting Imputation Model
    Parameters:
        params: dict | None = None,
        num_boost_round: int = 50
    Returns:
        pd.DataFrame: Imputed dataframe
    """
    def __init__(self, params: dict | None = None, num_boost_round: int = 500):
        self.params = params or {
            "objective": "reg:squarederror",
            "max_depth": 8,
            "eta": 0.05,
            "verbosity": 0,
            "tree_method": "hist",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "rmse",
            "nthread": -1,
            "yColumn": None,
        }
        self.num_boost_round = num_boost_round
        self.forward_model = None
        self.backward_model = None
        self.model = None

    def set_index(self, df: pd.DataFrame):
        """
        Set the index of the dataframe
        """
        date_column_names = [
            "date",
            "time",
            "datetime",
            "timestamp",
            "Date",
            "DATE",
            "Time",
            "TIME",
            "DateTime",
            "DATETIME",
            "Timestamp",
            "TIMESTAMP",
        ]

        date_column_found = False
        for col_name in date_column_names:
            if col_name in df.columns:
                df = df.set_index(col_name)
                df.index = pd.to_datetime(df.index)
                date_column_found = True
                break
        if not date_column_found:
            raise ValueError("Date column not found in dataframe")
        return df

    def create_features(self, df: pd.DataFrame, lags: int = 3, window: int = 3) -> pd.DataFrame:
        """
        Create features for the dataframe
        """
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df = self.set_index(df)
        df = df.copy()

        for i in range(1, lags + 1):
            df[f"lag_{i}"] = df[self.params["yColumn"]].shift(i)
        df[f"rolling_mean_{window}"] = df[self.params["yColumn"]].shift(1).rolling(window).mean()
        df["hour"] = df.index.hour # type: ignore
        df["dayofweek"] = df.index.dayofweek # type: ignore
        df["month"] = df.index.month # type: ignore
        return df

    def fit(self, df: pd.DataFrame):
        """
        Fit the forward and backward models to the dataframe
        """
        if self.params["yColumn"] is None:
            raise ValueError("yColumn must be specified")

        print('Training XGBoost regressors...\n', end='')
        df = self.set_index(df)

        # Forward pass
        first_nan_idx = df[self.params["yColumn"]].isna().idxmax()
        forward_df = df.loc[:first_nan_idx].dropna()
        forward_feat = self.create_features(forward_df).dropna()
        X_forward = forward_feat.drop(columns=[self.params["yColumn"]])
        y_forward = forward_feat[self.params["yColumn"]]
        dtrain_fwd = xgb.DMatrix(X_forward, label=y_forward)
        self.forward_model = xgb.train(
            self.params, dtrain_fwd, num_boost_round=self.num_boost_round
        )

        # Backward pass
        last_nan_idx = df[self.params["yColumn"]].isna()[::-1].idxmax()
        backward_df = df.loc[last_nan_idx:].dropna()
        backward_feat = self.create_features(backward_df).dropna()
        X_backward = backward_feat.drop(columns=[self.params["yColumn"]])
        y_backward = backward_feat[self.params["yColumn"]]
        dtrain_bwd = xgb.DMatrix(X_backward, label=y_backward)
        self.backward_model = xgb.train(
            self.params, dtrain_bwd, num_boost_round=self.num_boost_round
        )

    def _impute_pass(self, df: pd.DataFrame, model, direction="forward") -> list[float]:
        """
        Imputation pass for forward or backward direction
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = self.set_index(df)

        df = df.copy()
        updated_values = []

        for idx in df[df[self.params["yColumn"]].isna()].index:
            feat_df = self.create_features(df)
            row = feat_df.loc[[idx]].drop(columns=[self.params["yColumn"]])
            if row.isna().any(axis=1).values[0]:
                continue
            dmatrix = xgb.DMatrix(row)
            pred = model.predict(dmatrix)[0]
            df.at[idx, self.params["yColumn"]] = pred
            updated_values.append(pred)
        return updated_values

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute the dataframe
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = self.set_index(df)

        mask = df[self.params["yColumn"]].isnull().to_numpy()
        missing_values = df.loc[mask].copy()

        print('Imputing forward...')
        df_forward = self._impute_pass(df.copy(), model=self.forward_model, direction="forward")
        print('Imputing backward...')
        df_backward = self._impute_pass(df[::-1].copy(), model=self.backward_model, direction="backward")[::-1]

        missing_values[self.params["yColumn"]] = np.round(
            ((np.array(df_forward, dtype=float) + np.array(df_backward, dtype=float)) / 2), 2
        )
        missing_values.reset_index(inplace=True)
        return missing_values

    def evaluate(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """
        Evaluate the model
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = self.set_index(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        df_feat = self.create_features(df).dropna()
        X = df_feat.drop(columns=[self.params["yColumn"]])
        y = df_feat[self.params["yColumn"]]

        scores = {"MAE": [], "RMSE": []}

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)

            scores["MAE"].append(mean_absolute_error(y_test, y_pred))
            scores["RMSE"].append(mean_squared_error(y_test, y_pred, squared=False))

        print('Evaluated, returning MAE and RMSE')
        return {
            "MAE": np.mean(scores["MAE"]),
            "RMSE": np.mean(scores["RMSE"]),
        }


# df = pd.read_csv("/Users/npatil14/Downloads/IITC/Assignments/CS-597/regenSystem/datasets/AirPassengers.csv")
# model = GradientBoostingImputationModel(
#     {
#         "objective": "reg:squarederror",
#         "max_depth": 8,
#         "eta": 0.05,
#         "verbosity": 0,
#         "tree_method": "hist",
#         "subsample": 0.8,
#         "colsample_bytree": 0.8,
#         "min_child_weight": 5,
#         "eval_metric": "rmse",
#         "nthread": -1,
#         "yColumn": "Passengers",
#     }
# )
# model.fit(df)

# imputed_df = model.impute(df)
# print(imputed_df)
