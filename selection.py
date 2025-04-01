import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import os
from datetime import datetime
from scipy.stats import linregress


# ===============================
# Dataset Feature Extraction
# ===============================
def extract_features(df: pd.DataFrame) -> dict:
    """
    Extract key characteristics from the input DataFrame.

    Characteristics extracted:
    - length: number of rows in the DataFrame.
    - missing_rate: overall fraction of missing values (averaged across columns).
    - has_trend: binary indicator if a trend exists in the first numeric column.
    - has_seasonality: binary indicator based on inferred frequency or autocorrelation.

    Parameters:
    - df: pd.DataFrame containing the time-series data.

    Returns:
    - features: dict with keys 'length', 'missing_rate', 'has_trend', 'has_seasonality'
    """
    # Length of dataset
    length = len(df)

    # Overall missing rate (average fraction missing per column)
    missing_rate = df.isnull().mean().mean() if length > 0 else 0.0

    # Trend detection using linear regression on first numeric column
    # (Filling missing values by forward fill as a simple approach)
    first_numeric_col = df.select_dtypes(include=[np.number]).columns
    if len(first_numeric_col) > 0:
        col = first_numeric_col[0]
        # Fill missing values temporarily
        series = df[col].ffill().bfill()
        # Create time variable as integer sequence
        x = np.arange(length)
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, series)
        # Define a threshold for trend presence (absolute slope > small epsilon)
        has_trend = abs(slope) > 1e-3
    else:
        has_trend = False

    # Seasonality detection:
    # Check if the DataFrame index is a datetime index with an inferred frequency
    if isinstance(df.index, pd.DatetimeIndex):
        freq = pd.infer_freq(df.index)
        has_seasonality = freq is not None
    else:
        has_seasonality = False

    features = {
        "length": length,
        "missing_rate": missing_rate,
        "has_trend": int(has_trend),  # 1 if trend exists, else 0
        "has_seasonality": int(has_seasonality),  # 1 if seasonality is detected, else 0
    }
    return features


# ===============================
# Model Selection using Decision Tree
# ===============================
class ModelSelector:
    def __init__(self, candidate_models=None):
        """
        Initialize the ModelSelector with an empty training record and a decision tree regressor.

        Parameters:
        - candidate_models: Optional list of candidate model names.
          If None, defaults to a predefined list.
        """
        # Candidate imputation models; modify or expand this list as needed.
        self.candidate_models = candidate_models or [
            "knn_imputation",
            "regression_imputation",
            "mice_imputation",
            "spline_imputation",
            "lstm_imputation",
        ]

        # These lists store historical records
        self.X_train = []  # List of feature vectors
        self.y_train = (
            []
        )  # Best model index (as integer) for the corresponding feature vector

        # Initialize decision tree regressor
        self.tree = DecisionTreeRegressor(max_depth=5)

    def add_record(self, features: dict, best_model: str):
        """
        Add a record of a dataset's features and the best-performing candidate model.

        Parameters:
        - features: dict returned from extract_features().
        - best_model: string representing the best candidate model (must be in candidate_models list).
        """
        # Convert feature dict to a vector using a fixed order
        feature_vector = [
            features.get("length", 0),
            features.get("missing_rate", 0.0),
            features.get("has_trend", 0),
            features.get("has_seasonality", 0),
        ]
        self.X_train.append(feature_vector)
        try:
            model_idx = self.candidate_models.index(best_model)
        except ValueError:
            # If the provided model is not in the candidate list, default to first candidate.
            model_idx = 0
        self.y_train.append(model_idx)

    def train(self):
        """
        Train the decision tree regressor on the collected feature vectors and best model indices.
        """
        if self.X_train and self.y_train:
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            self.tree.fit(X, y)
        else:
            print("No training records available. Skipping training.")

    def predict(self, features: dict) -> str:
        """
        Predict the best candidate model for a new dataset based on its features.

        Parameters:
        - features: dict returned from extract_features().

        Returns:
        - selected_model: string, the recommended candidate model.
        """
        # Convert feature dict to vector
        feature_vector = np.array(
            [
                [
                    features.get("length", 0),
                    features.get("missing_rate", 0.0),
                    features.get("has_trend", 0),
                    features.get("has_seasonality", 0),
                ]
            ]
        )

        # If no training records are available, default to the first candidate.
        if not self.X_train:
            return self.candidate_models[0]

        pred_idx = int(round(self.tree.predict(feature_vector)[0]))
        # Ensure the index is within valid bounds
        pred_idx = max(0, min(pred_idx, len(self.candidate_models) - 1))
        return self.candidate_models[pred_idx]

    def save(self, file_path="model_selector.pkl"):
        """
        Save the ModelSelector instance to disk.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path="model_selector.pkl"):
        """
        Load a ModelSelector instance from disk if available; otherwise, return a new instance.
        """
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            return ModelSelector()


# testing the module standalone
if __name__ == "__main__":
    # Create a dummy time-series DataFrame with a date index and one numeric column.
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = np.random.randn(100).cumsum()  # cumulative sum to simulate a trend
    df_dummy = pd.DataFrame({"value": data}, index=dates)

    # Introduce some missing values randomly
    df_dummy.iloc[10:15] = np.nan
    df_dummy.iloc[50] = np.nan

    # Extract features
    features = extract_features(df_dummy)
    print("Extracted Features:", features)

    # Initialize ModelSelector, add a dummy record, train and predict
    selector = ModelSelector()
    # For demonstration, assume that for these features the best model was 'mice_imputation'
    selector.add_record(features, "mice_imputation")
    selector.train()
    recommended_model = selector.predict(features)
    print("Recommended Model:", recommended_model)
