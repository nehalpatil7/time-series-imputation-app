import os
import re
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
from typing import Optional, Dict, Any, Tuple

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dropout,
    Dense,
    GRU,
)
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import save_model, load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from sklearn.preprocessing import StandardScaler

class LSTMImputationModel:
    """
    LSTM-based imputation model for time series data
    Uses GRUs to learn patterns & impute missing values
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None, df: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        """
        Initialize the LSTM imputation model.
        Args:
            params: Optional dictionary of model parameters
        """
        self.params = params or {
            "epochs": 30,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "yColumn": None,
            "use_forward_backward": True
        }
        self.df = df
        self.model = None
        self.scaler = None
        self.save_path = save_path or "models/checkpoints"
        # if self.save_path:
        #     print(f"[LSTMImputationModel | __init__] Loading models from {self.save_path}")
        #     fwd_path = os.path.join(self.save_path, "model_forward.h5")
        #     bwd_path = os.path.join(self.save_path, "model_backward.h5")
        #     if os.path.isfile(fwd_path) and os.path.isfile(bwd_path):
        #         self.model_forward = self.load_models(fwd_path)
        #         self.model_backward = self.load_models(bwd_path)
        #         print(f"[LSTMImputationModel | __init__] Loaded models from {self.save_path}")
        #     else:
        #         os.makedirs(self.save_path, exist_ok=True)

    def extract_number(self, val):
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group(0)) if match else None

    def save_models(self, forward_path: str, backward_path: str):
        """Manually save both models to arbitrary locations."""
        save_model(
            self.model_forward, forward_path, overwrite=True, include_optimizer=True
        )
        save_model(
            self.model_backward, backward_path, overwrite=True, include_optimizer=True
        )
        print(f"[LSTMImputationModel | save_models] Saved models to {forward_path} and {backward_path}")

    # def load_models(
    #     self, forward_path: Optional[str] = None, backward_path: Optional[str] = None
    # ):
    #     """
    #     Load models from disk. If no paths are given, will use save_dir defaults.
    #     """
    #     if forward_path is None or backward_path is None:
    #         if not self.save_path:
    #             raise ValueError("No save_path specified and no explicit paths given")
    #         forward_path = forward_path or os.path.join(
    #             self.save_path, "model_forward.h5"
    #         )
    #         backward_path = backward_path or os.path.join(
    #             self.save_path, "model_backward.h5"
    #         )

    #     custom_objs = {"mse": MeanSquaredError()}

    #     if os.path.isfile(forward_path):
    #         self.model_forward = load_model(forward_path, custom_objects=custom_objs)
    #     else:
    #         print(f"No forward model at {forward_path}")

    #     if os.path.isfile(backward_path):
    #         self.model_backward = load_model(backward_path, custom_objects=custom_objs)
    #     else:
    #         print(f"No backward model at {backward_path}")

    #     print(
    #         f"[LSTMImputationModel | load_models] Loaded models:\n forward -> {forward_path}\n backward -> {backward_path}"
    #     )

    def prepare_sequences(self, df):
        """
        Prepare sequences for LSTM training/prediction.
        Args:
            is_forward: Whether to prepare forward or backward sequences
        Returns:
            Tuple of (X, y) sequences
        """
        if self.df is None:
            raise ValueError("Model must be fitted before transformation")

        X, y = [], []
        prediction_length = 1  # number of days to predict
        past_days = 14         # number of past days to consider
        y_col_idx = self.df.columns.get_loc(self.params["yColumn"])

        for i in range(past_days, len(df) - prediction_length + 1):
            X.append(df[i - past_days:i, 0:self.df.shape[1]])
            y.append(df[i + prediction_length - 1:i + prediction_length, y_col_idx])

        return np.array(X), np.array(y)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the input data and split into forward/backward sequences.
        Returns:
            Tuple of (forward_df, backward_df) or (None, None) if error occurs
        """
        try:
            if self.df is None:
                raise ValueError("[LSTMImputationModel | preprocess_data] Model must be fitted before transformation")

            df = self.df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = df[col].apply(self.extract_number)
                df[col] = df[col].astype(float)

            # Get null indices in the target column
            y_column = self.params["yColumn"]
            if y_column not in self.df.columns:
                raise ValueError(f"[LSTMImputationModel | preprocess_data] Target column {y_column} not found in DataFrame")

            mask = df[y_column].isnull().to_numpy()
            missing_locations = df.loc[mask].copy()
            missing_values = missing_locations.shape[0]
            if missing_values == 0:
                raise ValueError("[LSTMImputationModel | preprocess_data] No null values found in the target column")

            first_null_idx = missing_locations.index[0]
            last_null_idx = missing_locations.index[-1]

            # Get the indices before first null and after last null
            all_dates = df.index
            first_valid_idx = all_dates[all_dates < first_null_idx][-1]
            last_valid_idx = all_dates[all_dates > last_null_idx][0]

            forward_df = df.loc[:first_valid_idx].copy()
            backward_df = df.loc[last_valid_idx:].copy()
            forward_df, backward_df = [
                df.ffill().bfill() for df in (forward_df, backward_df)
            ]

            # Scale the data
            feature_cols = [c for c in df.columns if c != self.params["yColumn"]]
            self.scaler = StandardScaler()

            # SCALE FORWARD BLOCK
            Xf_feat = self.scaler.fit_transform(forward_df[feature_cols])
            yf = self.scaler.fit_transform(forward_df[[self.params["yColumn"]]])
            forward_scaled = np.hstack([yf, Xf_feat])

            # SCALE BACKWARD BLOCK
            Xb_feat = self.scaler.fit_transform(backward_df[feature_cols])
            yb = self.scaler.fit_transform(backward_df[[self.params["yColumn"]]])
            backward_scaled = np.hstack([yb, Xb_feat])

            forward_trainX, forward_trainY = self.prepare_sequences(forward_scaled)
            backward_trainX, backward_trainY = self.prepare_sequences(backward_scaled)

            return forward_trainX, forward_trainY, backward_trainX, backward_trainY
        except Exception as e:
            raise ValueError(f"[LSTMImputationModel | preprocess_data] Error in preprocess_data: {e}")

    def fit(self) -> None:
        """
        Fit the LSTM model to the training data.
        Args:
            df: Input DataFrame containing the time series data
        """
        forward_trainX, forward_trainY, backward_trainX, backward_trainY = self.preprocess_data()
        assert not np.isnan(forward_trainX).any(), "NaNs in forward X!"
        assert not np.isnan(forward_trainY).any(), "NaNs in forward Y!"
        assert not np.isnan(backward_trainX).any(), "NaNs in backward X!"
        assert not np.isnan(backward_trainY).any(), "NaNs in backward Y!"

        opt_fwd = Adam(
            learning_rate=self.params.get("learning_rate", 1e-3), clipnorm=1.0
        )
        opt_bwd = Adam(
            learning_rate=self.params.get("learning_rate", 1e-3), clipnorm=1.0
        )

        # Build the forward model
        self.model_forward = Sequential()
        self.model_forward.add(GRU(64, activation='relu', input_shape=(forward_trainX.shape[1], forward_trainX.shape[2]), return_sequences=True))
        self.model_forward.add(GRU(32, activation="relu", return_sequences=True))
        self.model_forward.add(Dropout(self.params["dropout_rate"]))
        self.model_forward.add(Dense(forward_trainY.shape[1]))
        self.model_forward.compile(optimizer=opt_fwd, loss='mse')
        self.model_forward.fit(
            forward_trainX,
            forward_trainY,
            epochs=self.params["epochs"],
            batch_size=16,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            validation_split=0.1,
            verbose=0
        )

        # Build the backward model
        self.model_backward = Sequential()
        self.model_backward.add(GRU(64, activation='relu', input_shape=(backward_trainX.shape[1], backward_trainX.shape[2]), return_sequences=True))
        self.model_backward.add(GRU(32, activation="relu", return_sequences=True))
        self.model_backward.add(Dropout(self.params["dropout_rate"]))
        self.model_backward.add(Dense(backward_trainY.shape[1]))
        self.model_backward.compile(optimizer=opt_bwd, loss="mse", run_eagerly=True)
        self.model_backward.fit(
            backward_trainX,
            backward_trainY,
            epochs=self.params["epochs"],
            batch_size=16,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            validation_split=0.1,
            verbose=0
        )

        # os.makedirs(self.save_path, exist_ok=True)
        # self.save_models(os.path.join(self.save_path, "model_forward.h5"), os.path.join(self.save_path, "model_backward.h5"))

    def impute(self, only_missing_dates: bool = False) -> pd.DataFrame:
        """
        Impute missing values using the trained LSTM model.
        Returns:
            DataFrame with imputed values
        """
        if self.df is None:
            raise ValueError("[LSTMImputationModel | impute] Model must be fitted before prediction")

        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        y_col = self.params["yColumn"]

        mask = df[y_col].isnull()
        missing_locations = df.index[mask]
        M = len(missing_locations)

        if y_col in ["price", "Open", "High", "Low", "Close"]:
            missing_dates = pd.date_range(
                start=df.index[mask].min(),
                periods=M,
                freq=us_bd,
                tz="UTC",
            )
        else:
            missing_dates = pd.date_range(
                start=df.index[mask].min(), periods=M, tz="UTC"
            )

        # Make predictions
        fwd_X, _, bwd_X, _ = self.preprocess_data()
        fwd_pred = self.model_forward.predict(fwd_X[-min(M, fwd_X.shape[0]) :])
        bwd_pred = self.model_backward.predict(bwd_X[::-1][-min(M, bwd_X.shape[0]) :])

        # Inverse transform predictions
        if self.scaler is None:
            raise ValueError("[LSTMImputationModel | impute] Scaler not found. Please fit the model first.")
        fwd_yScaled = fwd_pred[:, -1, 0].reshape(-1, 1)
        fwd_yPred = self.scaler.inverse_transform(fwd_yScaled).flatten() # type: ignore
        preds = pd.DataFrame({"fwd_yPred": pd.Series(fwd_yPred)}, index=missing_dates)

        bwd_yScaled = bwd_pred[:, -1, 0].reshape(-1, 1)
        bwd_yPred = self.scaler.inverse_transform(bwd_yScaled).flatten()[::-1] # type: ignore
        preds = preds.assign(bwd_yPred=pd.Series(bwd_yPred[::-1]))
        preds.to_csv("preds.csv")

        # Create DataFrame with imputed values
        imputed_values = (fwd_yPred + bwd_yPred) / 2
        missing_df = pd.DataFrame({
            'Date': missing_dates,
            y_col: imputed_values
        }).set_index('Date')

        if only_missing_dates:
            missing_df = missing_df.reindex(missing_locations).bfill().ffill()
            return missing_df.reset_index()

        df.loc[missing_dates, y_col] = imputed_values
        df = df.bfill().ffill()
        return df.reset_index()

df = pd.read_csv("/Users/npatil14/Downloads/IITC/Assignments/CS-597/regenSystem/datasets/btc.csv")
model = LSTMImputationModel({"yColumn": "price", "epochs": 10, "dropout_rate": 0.2, "learning_rate": 0.01}, df=df)
model.fit()
result = model.impute()