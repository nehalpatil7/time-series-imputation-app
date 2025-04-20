import numpy as np
import pandas as pd
from sklearn.tree import (
    DecisionTreeRegressor,
)
import pickle
import os
import logging
from datetime import datetime
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='imputation.log',
    filemode='a'
)
logger = logging.getLogger('imputation')

# Set plotting style
plt.style.use("ggplot")
sns.set(font_scale=1.2)


def extract_features(df: pd.DataFrame) -> dict:
    """
    Extract key characteristics from the input DataFrame relevant for imputation.
    Characteristics extracted:
    - length: number of rows in the DataFrame.
    - missing_rate: overall fraction of missing values.
    - has_trend: binary indicator if a trend exists in the first numeric column.
    - has_seasonality: binary indicator based on inferred frequency.
    - freq: inferred frequency of the time series data (for ARIMA, etc.)
    - order: suggested ARIMA order parameters (p,d,q)
    - seasonal_order: suggested seasonal order parameters (P,D,Q,s)
    Parameters:
    - df: pd.DataFrame containing the time-series data.
    Returns:
    - features: dict with extracted features and model parameters
    """
    # Length of dataset
    length = len(df)

    # Overall missing rate (average fraction missing per column)
    missing_rate = df.isnull().mean().mean() if length > 0 else 0.0

    # Trend detection using linear regression on first numeric column
    first_numeric_col = df.select_dtypes(include=[np.number]).columns
    has_trend = False
    if (
        len(first_numeric_col) > 0 and length > 1
    ):  # Need at least 2 points for regression
        col = first_numeric_col[0]
        # Fill missing values temporarily for trend analysis only
        series = df[col].ffill().bfill()
        if (
            not series.isnull().all() and len(series.dropna()) > 1
        ):  # Check if any non-null values remain
            x = np.arange(len(series))
            # Perform linear regression only on non-missing points after filling
            valid_indices = ~series.isnull()
            if valid_indices.sum() > 1:
                # Ultra-simple trend detection - just check first vs last value
                valid_values = series[valid_indices].tolist()  # Convert to list to avoid array typing issues
                valid_times = x[valid_indices].tolist()

                if len(valid_values) >= 2:
                    # Compare first and last values
                    first_value = valid_values[0]
                    last_value = valid_values[-1]
                    first_time = valid_times[0]
                    last_time = valid_times[-1]

                    # Only calculate slope if time difference exists
                    if last_time > first_time:
                        # Simple slope calculation
                        slope = (last_value - first_value) / (last_time - first_time)
                        # Check if the trend is significant relative to the data range
                        data_range = max(valid_values) - min(valid_values)
                        if data_range > 0:
                            relative_change = abs(last_value - first_value) / data_range
                            has_trend = abs(slope) > 1e-5 and relative_change > 0.1
                        else:
                            has_trend = False
                    else:
                        has_trend = False
                else:
                    has_trend = False
            else:
                has_trend = False  # Not enough points for trend
        else:
            has_trend = False  # All values were NaN or only one remained

    # Seasonality detection and frequency inference
    has_seasonality = False
    inferred_freq = None
    seasonal_period = 0

    # Check if there's a date column we can use if index is not datetime
    date_col = None
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check common date column names
        date_column_names = ['date', 'time', 'datetime', 'timestamp', 'Date', 'Time', 'DateTime', 'Timestamp']
        for col_name in date_column_names:
            if col_name in df.columns:
                try:
                    # Create a copy to avoid modifying the original dataframe
                    date_series = pd.to_datetime(df[col_name], errors='coerce')
                    if not date_series.isnull().all():
                        date_col = date_series
                        break
                except Exception as e:
                    logger.warning(f"Failed to convert column {col_name} to datetime: {e}")

    # Use either the index or found date column
    date_index = df.index if isinstance(df.index, pd.DatetimeIndex) else date_col

    if date_index is not None:
        try:
            # infer_freq can be computationally intensive on large irregular series
            if length < 100000:
                try:
                    # For datetime index
                    if isinstance(date_index, pd.DatetimeIndex):
                        inferred_freq = pd.infer_freq(date_index)
                    # For datetime column
                    elif date_col is not None:
                        # Use only non-null values to create a proper DatetimeIndex
                        clean_dates = date_col.dropna()
                        if len(clean_dates) > 1:
                            # Create a DatetimeIndex from clean dates for frequency inference
                            date_index = pd.DatetimeIndex(clean_dates)
                            inferred_freq = pd.infer_freq(date_index)
                    else:
                        inferred_freq = None

                    has_seasonality = inferred_freq is not None

                    # Determine seasonal period based on frequency
                    if inferred_freq:
                        # Map common frequency strings to their seasonal periods
                        freq_mapping = {
                            'D': 7,       # Daily -> Weekly seasonality
                            'B': 5,       # Business daily -> Weekly (5 days)
                            'W': 52,      # Weekly -> Yearly
                            'M': 12,      # Monthly -> Yearly
                            'Q': 4,       # Quarterly -> Yearly
                            'A': 1,       # Annual -> No higher seasonality
                            'H': 24,      # Hourly -> Daily
                            'min': 60,    # Minute -> Hourly
                            'T': 60,      # Minute -> Hourly
                            'S': 60       # Second -> Minute
                        }

                        # Extract base frequency (first character for most freq strings)
                        base_freq = inferred_freq[0] if inferred_freq else ''
                        seasonal_period = freq_mapping.get(base_freq, 0)

                        # Special case for hourly data with clear daily pattern
                        if base_freq in ['H', 'min', 'T', 'S'] and length >= 48:  # At least 2 days of hourly data
                            has_seasonality = True
                except Exception as e:
                    logger.warning(f"{datetime.now()} - Error inferring frequency: {e}")
            else:
                logger.warning(f"{datetime.now()} - Skipping frequency inference due to large dataset size.")
                # Could add basic seasonality check via ACF here if needed
                has_seasonality = False  # Defaulting for very large datasets
        except Exception as e:
            logger.warning(f"{datetime.now()} - Error during frequency detection: {e}")
            has_seasonality = False  # Handle potential errors

    # Calculate suggested ARIMA parameters based on data characteristics
    # Default values
    p, d, q = 1, 1, 1  # Default ARIMA order
    P, D, Q, s = 0, 0, 0, 0  # Default seasonal order

    # Differencing term (d) based on trend
    if has_trend:
        d = 1  # Differencing to remove trend
    else:
        d = 0

    # Seasonal components if seasonality detected
    if has_seasonality and seasonal_period > 0:
        P, D, Q, s = 1, 1, 1, seasonal_period

    # Create the features dictionary with all extracted information
    features = {
        "length": length,
        "missing_rate": missing_rate,
        "has_trend": int(has_trend),              # 1 if trend exists, else 0
        "has_seasonality": int(has_seasonality),  # 1 if seasonality is detected, else 0
        "freq": inferred_freq,                    # Pandas frequency string
        "seasonal_period": seasonal_period,       # Numeric seasonal period
        "arima_params": {
            "order": (p, d, q),                   # ARIMA order parameters
            "seasonal_order": (P, D, Q, s)        # Seasonal ARIMA parameters
        }
    }

    return features


class ImputationModelSelector:
    def __init__(self, candidate_models=None):
        """
        Initialize the ModelSelector for imputation models.
        Parameters:
        - candidate_models: Optional list of candidate imputation model names.
        """
        # Updated list focusing on imputation models
        self.candidate_models = candidate_models or [
            "linear_interpolation",
            "spline_interpolation",
            "mean_imputation",
            "knn_imputation",
            "regression_imputation",
            "mice_imputation",
            "arima_imputation",
            "gb_imputation",
            "lstm_imputation",
            "exponential_smoothing"
        ]

        self.X_train = []
        self.y_train = (
            []
        )

        self.tree = DecisionTreeRegressor(max_depth=5)

    def add_record(self, features: dict, best_model: str):
        """
        Add a record of a dataset's features and the best-performing imputation model.
        Parameters:
        - features: dict returned from extract_features().
        - best_model: string representing the best candidate model (must be in candidate_models list).
        """
        feature_vector = [
            features.get("length", 0),
            features.get("missing_rate", 0.0),
            features.get("has_trend", 0),
            features.get("has_seasonality", 0),
        ]

        self.X_train.append(feature_vector)

        try:
            model_idx = self.candidate_models.index(best_model)
            self.y_train.append(model_idx)
        except ValueError:
            logger.warning(f"{datetime.now()} - Imputation model '{best_model}' not found in candidate list. Defaulting to index 0.")
            self.y_train.append(0)  # Default to first model if not found

    def train(self):
        """
        Train the decision tree regressor on the collected feature vectors and best model indices.
        """
        if self.X_train and self.y_train:
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            # Ensure variability in target if using certain tree parameters
            if len(np.unique(y)) > 1:
                self.tree.fit(X, y)
            else:
                logger.warning(f"{datetime.now()} - Only one type of model recorded in training data. Tree cannot be effectively trained.")
                # In this case, prediction will likely just return the single recorded model index
                # Or handle this case specifically if needed (e.g., use rule-based anyway)
                self.tree = DecisionTreeRegressor(max_depth=1)
                self.tree.fit(X, y)
        else:
            logger.warning(f"{datetime.now()} - No training records available. Skipping training.")

    def _rule_based_selection(self, features: dict) -> str:
        """
        Select imputation model based on characteristics when no training data exists.
        Rules prioritize handling time series patterns and missingness levels.

        Parameters:
        - features: dict with time series characteristics

        Returns:
        - selected_model: string, recommended imputation model based on rules
        """
        length = features.get("length", 0)
        missing_rate = features.get("missing_rate", 0.0)
        has_trend = features.get("has_trend", 0)
        has_seasonality = features.get("has_seasonality", 0)

        available = set(self.candidate_models)

        # --- Rule Logic ---

        # Rule 1: Very low missing rate - simple methods often suffice
        if missing_rate < 0.05 and "linear_interpolation" in available:
            return "linear_interpolation"

        # Rule 2: Data with Trend and Seasonality
        if has_trend and has_seasonality:
            if length > 500 and "lstm_imputation" in available:
                return "lstm_imputation"  # Good for long, complex patterns
            elif "arima_imputation" in available:
                return "arima_imputation"  # Captures time dependencies
            elif "spline_interpolation" in available:
                return "spline_interpolation"  # Can handle curves better than linear

        # Rule 3: Data with only Trend
        elif has_trend and not has_seasonality:
            if "arima_imputation" in available:
                return "arima_imputation"  # ARIMA handles trend well
            elif "regression_imputation" in available:
                return "regression_imputation"  # Can model trend explicitly
            elif "linear_interpolation" in available:
                return "linear_interpolation"  # Simple for trend if gaps small

        # Rule 4: Data with only Seasonality
        elif not has_trend and has_seasonality:
            if "arima_imputation" in available:
                return "arima_imputation"  # Seasonal ARIMA part
            elif "spline_interpolation" in available:
                return "spline_interpolation"  # Can capture periodic shapes
            elif "knn_imputation" in available:
                return "knn_imputation"  # Might capture local seasonal patterns

        # Rule 5: Data with neither Trend nor Seasonality (or stationary)
        elif not has_trend and not has_seasonality:
            if missing_rate > 0.2 and "mice_imputation" in available:
                return "mice_imputation"  # Good for complex missingness, multivariate
            elif "knn_imputation" in available:
                return "knn_imputation"  # Good general purpose
            elif "mean_imputation" in available:
                return "mean_imputation"  # Simple baseline

        # Rule 6: High missing rate - favor robust methods
        if missing_rate > 0.3:
            if "mice_imputation" in available:
                return "mice_imputation"
            elif "knn_imputation" in available:
                return "knn_imputation"
            elif "gb_imputation" in available:  # Boosting can handle complex relations
                return "gb_imputation"

        # Rule 7: Very long sequences - consider advanced methods
        if length > 1000:
            if "lstm_imputation" in available:
                return "lstm_imputation"
            elif "gb_imputation" in available:
                return "gb_imputation"

        # Default Fallback (if no specific rules match strongly)
        if "knn_imputation" in available:
            return "knn_imputation"  # Often a reasonable default
        elif "linear_interpolation" in available:
            return "linear_interpolation"
        else:
            return self.candidate_models[0]  # Absolute fallback

    def predict(self, features: dict) -> str:
        """
        Predict the best candidate imputation model for a new dataset based on its features.
        Parameters:
        - features: dict returned from extract_features().
        Returns:
        - selected_model: string, the recommended candidate imputation model.
        """
        feature_vector = np.array(
            [
                features.get("length", 0),
                features.get("missing_rate", 0.0),
                features.get("has_trend", 0),
                features.get("has_seasonality", 0),
            ]
        ).reshape(1, -1)

        # If no training records are available OR tree wasn't trained, use rule-based selection
        if not self.X_train or not hasattr(self.tree, 'fit'):
            logger.info(f"{datetime.now()} - Using rule-based selection for recommendation.")
            return self._rule_based_selection(features)

        try:
            pred_idx = int(round(self.tree.predict(feature_vector)[0]))
            pred_idx = max(0, min(pred_idx, len(self.candidate_models) - 1))
            logger.info(f"{datetime.now()} - Using trained Decision Tree for recommendation.")
            return self.candidate_models[pred_idx]
        except Exception as e:
            logger.error(f"{datetime.now()} - Error during prediction with trained tree: {e}. Falling back to rule-based selection.")
            return self._rule_based_selection(features)

    def save(self, file_path="imputation_model_selector.pkl"):
        """Save the ImputationModelSelector instance to disk."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path="imputation_model_selector.pkl"):
        """Load an ImputationModelSelector instance from disk if available; otherwise, return a new instance."""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            return ImputationModelSelector()
