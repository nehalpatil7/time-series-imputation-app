# models/linear_interpolation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer


class LinearInterpolationModel:
    def __init__(self):
        pass

    def impute(self, df, method="linear"):
        """
        Perform linear interpolation on the dataframe.
        Returns only the imputed values with their corresponding dates.
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Interpolation method to use
        Returns:
            pd.DataFrame: DataFrame containing only the imputed values with dates
        """
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Ensure the index is datetime if it's not already
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            if 'Date' in df_copy.columns:
                df_copy.set_index('Date', inplace=True)
            else:
                # If no Date column, create a datetime index
                df_copy.index = pd.to_datetime(df_copy.index)

        missing_mask = df_copy.isna()

        # Perform interpolation
        df_copy.interpolate(method=method, limit_direction="forward", limit_area="inside", inplace=True)

        # Extract only the imputed values
        imputed_values = df_copy.where(missing_mask)
        missing_dates = df_copy.index[missing_mask.any(axis=1)]

        y_col = df_copy.select_dtypes(include=[np.number]).columns[0]

        result = pd.DataFrame({
            'Date': missing_dates,
            y_col: imputed_values.loc[missing_dates, y_col].round(2)
        })

        return result

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout, method="linear")
        # Evaluate on the first numeric column.
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}


# df = pd.read_csv("/Users/npatil14/Downloads/IITC/Assignments/CS-597/regenSystem/datasets/AirPassengers.csv")
# model = LinearInterpolationModel()
# result = model.impute(df)
# result.to_csv("AirPassengers_imputed.csv", index=False)