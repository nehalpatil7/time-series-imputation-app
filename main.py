from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import uuid
import os
import pandas as pd
import numpy as np
import io
import re
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from selection import extract_features, ImputationModelSelector
from models import (
    LinearInterpolationModel,
    SplineImputationModel,
    ExponentialSmoothingModel,
    ARIMAImputationModel,
    KNNImputationModel,
    RegressionImputationModel,
    MICEImputationModel,
    GradientBoostingImputationModel,
    LSTMImputationModel,
)


app = FastAPI(title="Time Series Imputation API", version="0.1")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PERFORMANCE_FILE = "model_performance_records.json"
datasets = {}                   # dataset_id -> file path or data object
preprocessed_data = {}          # dataset_id -> {"data": ..., "features": ...}
model_performance_records = {}  # dataset_id -> evaluation metrics

model_selector = None           # Initialize model_selector on-demand to prevent auto-loading during import


# save model performance records to JSON
def save_model_performance_records():
    """
    Save the model performance records to a JSON file.
    """
    try:
        with open(MODEL_PERFORMANCE_FILE, 'w') as f:
            json.dump(model_performance_records, f)
        print(f"{datetime.now()} - Model performance records saved to {MODEL_PERFORMANCE_FILE}")
    except Exception as e:
        print(f"{datetime.now()} - Error saving model performance records: {str(e)}")


# load model performance records from JSON
def load_model_performance_records():
    """
    Load the model performance records from a JSON file.
    """
    global model_performance_records
    try:
        if os.path.exists(MODEL_PERFORMANCE_FILE):
            with open(MODEL_PERFORMANCE_FILE, 'r') as f:
                model_performance_records = json.load(f)
            print(f"{datetime.now()} - Model performance records loaded from {MODEL_PERFORMANCE_FILE}")
        else:
            print(f"{datetime.now()} - No existing model performance records file found. Starting with empty records.")
    except Exception as e:
        print(f"{datetime.now()} - Error loading model performance records: {str(e)}")

load_model_performance_records()


# ---------------------------
# Pydantic models for endpoints
# ---------------------------
class PreprocessRequest(BaseModel):
    dataset_id: str


class SelectModelRequest(BaseModel):
    dataset_id: str
    override_model: Optional[str] = None


class ImputeRequest(BaseModel):
    dataset_id: str
    selected_model: str
    selected_y_column: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None


class UpdateFeedbackRequest(BaseModel):
    dataset_id: str
    model_performance: Dict[str, Any]
    user_feedback: Optional[str] = None


def extract_number(val):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(match.group(0)) if match else None


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def test_app():
    return {"Congratulations": "Regeneration APIs LIVE!"}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file and return a unique dataset_id.
    """
    dataset_id = str(uuid.uuid4())
    file_location = f"uploaded_datasets/{dataset_id}_{file.filename}"
    os.makedirs("uploaded_datasets", exist_ok=True)

    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    datasets[dataset_id] = file_location
    return {"dataset_id": dataset_id, "filename": file.filename}


@app.post("/preprocess")
async def preprocess_dataset(request: PreprocessRequest):
    """
    Load the dataset using dataset_id, preprocess it, and extract feature vector.
    """
    dataset_id = request.dataset_id
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Try to infer datetime columns during loading
        df = pd.read_csv(
            datasets[dataset_id],
            parse_dates=True,
            infer_datetime_format=True,
            na_values=["NaN", "nan", "NA", "null", ""],
        )

        # Check for common date column names and set as index if found
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
                try:
                    df[col_name] = pd.to_datetime(df[col_name])
                    df.set_index(col_name, inplace=True)
                    break
                except Exception as e:
                    print(
                        f"{datetime.now()} - Warning: Failed to convert column {col_name} to datetime: {e}"
                    )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e}")

    # Extract features using the selection module function
    features = extract_features(df)

    # Store preprocessed data and features in memory
    preprocessed_data[dataset_id] = {"data": df, "features": features}

    return {
        "dataset_id": dataset_id,
        "features": features,
        "message": "Preprocessing complete",
    }


@app.post("/select_model")
async def select_model(request: SelectModelRequest):
    """
    Select the best candidate imputation model based on dataset features.
    If override_model is provided, return that model.
    """
    global model_selector

    dataset_id = request.dataset_id
    print(f"{datetime.now()} - Preprocessing dataset {dataset_id} for model selection")
    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Preprocessed data not found for this dataset"
        )

    features = preprocessed_data[dataset_id]["features"]

    # If the user provided an override, return that
    if request.override_model:
        selected_model = request.override_model
    else:
        # Initialize model_selector if not already
        if model_selector is None:
            model_selector = ImputationModelSelector.load()

        # Use the model_selector to predict the best candidate based on features
        selected_model = model_selector.predict(features)

    # Provide some initial metrics based on features
    basic_metrics = {
        "dataset_length": features.get("length", 0),
        "missing_rate": f"{features.get('missing_rate', 0) * 100:.2f}%",
        "estimated_quality": "Please run imputation to get actual metrics",
        "model_confidence": (
            "Auto-selected" if not request.override_model else "User-specified"
        ),
    }

    return {
        "dataset_id": dataset_id,
        "selected_model": selected_model,
        "metrics": basic_metrics,
    }


@app.get("/evaluate")
async def evaluate_dataset(dataset_id: str, metric: Optional[str] = None):
    """
    Retrieve evaluation metrics for the imputation performed on the dataset.
    """
    if dataset_id not in model_performance_records:
        raise HTTPException(
            status_code=404, detail="Evaluation metrics not found for this dataset"
        )

    metrics = model_performance_records[dataset_id]
    if metric:
        return {metric: metrics.get(metric, "Metric not found")}
    return {"dataset_id": dataset_id, "evaluation_metrics": metrics}


@app.post("/update_feedback")
async def update_feedback(request: UpdateFeedbackRequest):
    """
    Update the lookup matrix and decision tree with new model performance data and user feedback.
    """
    global model_selector

    dataset_id = request.dataset_id
    # In a real implementation, update the model_selector with the performance data:
    # For demonstration, assume we consider the current dataset features and performance to indicate the best model.
    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Dataset not found for feedback update"
        )

    if model_selector is None:
        model_selector = ImputationModelSelector.load()

    features = preprocessed_data[dataset_id]["features"]
    # use the performance data to decide the best model
    best_model = request.model_performance.get(
        "best_model", model_selector.candidate_models[0]
    )
    model_selector.add_record(features, best_model)
    model_selector.train()
    model_selector.save()

    # Update model performance records JSON with user feedback if provided
    if dataset_id in model_performance_records and request.user_feedback:
        if "user_feedback" not in model_performance_records[dataset_id]:
            model_performance_records[dataset_id]["user_feedback"] = []
        model_performance_records[dataset_id]["user_feedback"].append({
            "feedback": request.user_feedback,
            "timestamp": datetime.now().isoformat()
        })
        save_model_performance_records()

    return {"dataset_id": dataset_id, "status": "Feedback updated successfully"}


@app.get("/dataset_columns/{dataset_id}")
async def get_dataset_columns(dataset_id: str):
    """
    Retrieve the column names from the dataset for axis selection.
    Also check if dataset has a required 'Date' column.
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = preprocessed_data[dataset_id]["data"]

    columns = df.columns.tolist()
    index_name = df.index.name if df.index.name else "Index"
    has_date_column = "Date" in columns

    if index_name == "Date" and "Date" not in columns:
        df = preprocessed_data[dataset_id]["data"].copy()
        df["Date"] = df.index
        preprocessed_data[dataset_id]["data"] = df
        columns.append("Date")
        has_date_column = True

    column_types = {}
    for col in columns:
        if col == "Date":
            column_types[col] = "datetime"
            try:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(
                    df[col]
                ):
                    df = preprocessed_data[dataset_id]["data"].copy()
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    preprocessed_data[dataset_id]["data"] = df
            except Exception as e:
                print(
                    f"{datetime.now()} - Warning: Failed to convert Date column to datetime: {e}"
                )

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
            df[col]
        ):
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors="raise")
                    column_types[col] = "datetime"
                else:
                    column_types[col] = "categorical"
            except (TypeError, ValueError):
                df[col] = df[col].apply(extract_number)
                df[col] = df[col].apply(pd.to_numeric, errors="ignore")
                numeric_cols = pd.api.types.is_numeric_dtype(df[col])
                if numeric_cols:
                    column_types[col] = "numeric"
                else:
                    column_types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        else:
            df[col] = df[col].apply(extract_number)
            df[col] = df[col].apply(pd.to_numeric, errors='ignore')
            numeric_cols = pd.api.types.is_numeric_dtype(df[col])
            if numeric_cols:
                column_types[col] = "numeric"
            else:
                column_types[col] = "categorical"

    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

    return {
        "columns": columns,
        "column_types": column_types,
        "index_name": index_name,
        "has_datetime_index": has_datetime_index,
        "has_date_column": has_date_column,
    }


@app.get("/dataset/{dataset_id}")
async def get_dataset(
    dataset_id: str, x_column: Optional[str] = None, y_column: Optional[str] = None
):
    """
    Retrieve the dataset for visualization with information about missing values.
    Preferred column for X-axis is 'Date', and Y-axis should be a numeric column.
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = preprocessed_data[dataset_id]["data"].copy()
    # Replace string NaN values with actual NaN
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].replace(["NaN", "nan", "NA", "N/A", "null", ""], np.nan)

    # Always prefer 'Date' as the X column
    use_index_as_x = True
    x_is_datetime = False

    # First check if 'Date' column exists
    if "Date" in df.columns:
        x_column = "Date"
        use_index_as_x = False
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                x_is_datetime = True
            except (ValueError, TypeError):
                x_is_datetime = False
        else:
            x_is_datetime = True

        x_series = df["Date"]
    elif x_column is not None and x_column in df.columns:
        use_index_as_x = False
        if pd.api.types.is_string_dtype(df[x_column]) or pd.api.types.is_object_dtype(
            df[x_column]
        ):
            try:
                df[x_column] = pd.to_datetime(df[x_column], errors="coerce")
                x_is_datetime = True
            except (ValueError, TypeError):
                x_is_datetime = False
        else:
            x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_column])

        x_series = df[x_column]
    else:
        x_is_datetime = isinstance(df.index, pd.DatetimeIndex)

    # Determine Y column (default to first numeric column if not specified)
    if y_column is not None and y_column in df.columns:
        column = y_column
    else:
        for col in df.columns:
            if x_column is not None and col.lower() not in x_column.lower():
                df[col] = df[col].apply(extract_number)
                df[col] = df[col].apply(pd.to_numeric, errors='ignore')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise HTTPException(
                        status_code=400, detail="No numeric columns found in dataset"
                    )
                column = numeric_cols[0]

    data_points = []
    numeric_index = range(len(df))
    missing_count = df[column].isna().sum()
    print(f"{datetime.now()} - Column {column} has {missing_count} missing values")

    for i, (idx, row) in enumerate(df.iterrows()):
        if use_index_as_x:
            if x_is_datetime and hasattr(idx, "strftime"):
                x_value = idx.strftime("%Y-%m-%d %H:%M:%S")
            else:
                x_value = numeric_index[i]
        else:
            x_val = x_series.iloc[i]
            if pd.api.types.is_datetime64_any_dtype(x_series.dtype):
                x_value = (
                    x_val.strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(x_val) and hasattr(x_val, "strftime")
                    else None
                )
            elif pd.api.types.is_numeric_dtype(x_series.dtype):
                x_value = float(x_val) if pd.notna(x_val) else None
            else:
                x_value = str(x_val) if pd.notna(x_val) else None

        # Check if Y value missing
        is_missing = pd.isna(row[column])

        if pd.isna(row[column]):
            y_value = None
        elif pd.api.types.is_numeric_dtype(type(row[column])):
            y_value = float(row[column])
        else:
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(row[column]))
            y_value = float(match.group(0)) if match else None

        data_points.append({"x": x_value, "y": y_value, "missing": is_missing})

    return {
        "column_name": column,
        "x_column": x_column if not use_index_as_x else "index",
        "data": data_points,
        "index_type": "datetime" if x_is_datetime else "numeric",
    }


@app.post("/impute")
async def impute_dataset(request: ImputeRequest):
    """
    Impute the missing values in the dataset using the selected model.
    For ARIMA models, additional parameters can be provided.
    """
    dataset_id = request.dataset_id
    selected_model = request.selected_model
    model_params = request.model_params
    y_column = request.selected_y_column

    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Preprocessed data not found for this dataset"
        )

    # Load the data
    df = preprocessed_data[dataset_id]["data"]

    print(f"Imputing with model: {selected_model}")
    def get_model_by_name(model_name: str):
        model_mapping = {
            # "linear_interpolation": LinearInterpolationModel(),
            # "spline_interpolation": SplineImputationModel(),
            "mean_imputation": lambda df: df.fillna(df.mean()),
            # "knn_imputation": KNNImputationModel(),
            # "regression_imputation": RegressionImputationModel(),
            # "mice_imputation": MICEImputationModel(),
            "arima_imputation": ARIMAImputationModel(),
            "gb_imputation": GradientBoostingImputationModel(),
            "lstm_imputation": LSTMImputationModel(),
            # "exponential_smoothing": ExponentialSmoothingModel(),
        }
        return model_mapping.get(model_name.lower())

    imputer = get_model_by_name(selected_model)
    print(f"[IMPUTE] Selected imputer: {imputer.__class__.__name__ if imputer else None}")

    if imputer is None:
        if selected_model == "mean_imputation":
            imputed_data = df.fillna(df.mean())
        else:
            imputed_data = df.fillna(method="ffill").fillna(method="bfill")
    elif callable(imputer) and not hasattr(imputer, "fit"):
        imputed_data = imputer(df)
    else:
        try:
            if imputer.__class__.__name__ == "ARIMAImputationModel" and model_params:
                order = tuple(model_params.get("order", (1, 1, 1)))
                seasonal_order = tuple(model_params.get("seasonal_order", (1, 1, 1, 1)))
                y_column = model_params.get("y_column", None)
                hasSeasonality = model_params.get("hasSeasonality", False)
                imputer = ARIMAImputationModel(
                    order=order,
                    seasonal_order=seasonal_order,
                    y_column=y_column,
                    use_forward_backward=True,
                    seasonality=hasSeasonality
                )
                imputed_data = imputer.transform(df)
            elif imputer.__class__.__name__ == "GradientBoostingImputationModel":
                imputer = GradientBoostingImputationModel(
                    {
                        "objective": "reg:squarederror",
                        "max_depth": 32 if len(df) > 10000 else 8,
                        "eta": 0.01 if len(df) > 10000 else 0.05,
                        "grow_policy": "depthwise" if len(df) > 10000 else "lossguide",
                        "verbosity": 0,
                        "tree_method": "hist",
                        "subsample": 0.8,
                        "early_stopping_rounds": 100 if len(df) > 10000 else 50,
                        "colsample_bytree": 0.8,
                        "max_leaves": 128 if len(df) > 10000 else 64,
                        "min_child_weight": 5,
                        "eval_metric": "rmse",
                        "nthread": -1,
                        "yColumn": y_column,
                    }
                )
                imputer.fit(df)
                imputed_data = imputer.impute(df)
            elif imputer.__class__.__name__ == "LSTMImputationModel":
                if model_params:
                    imputer = LSTMImputationModel(
                        params={
                            "epochs": 10,
                            "dropout_rate": 0.2,
                            "learning_rate": 0.001,
                            "yColumn": y_column,
                            "use_forward_backward": True
                        },
                        df=df,
                        save_path=f"models/checkpoints"
                    )
                imputer.fit()
                imputed_data = imputer.impute()
            else:
                train_df = df.copy()
                imputer.fit(train_df)
                imputed_data = imputer.transform(df)
        except Exception as e:
            print(
                f"{datetime.now()} - Error using {selected_model}: {str(e)}. Falling back to basic imputation."
            )
            imputed_data = df.fillna(method="ffill").fillna(method="bfill")

    try:
        eval_df = df.copy()
        non_null_mask = ~eval_df.isnull()

        import random
        np.random.seed(42)

        eval_metrics = {}
        for col in eval_df.select_dtypes(include=[np.number]).columns:
            non_null_indices = np.where(non_null_mask[col])[0]

            if len(non_null_indices) > 10:
                # Select 10% random indices
                test_size = max(int(len(non_null_indices) * 0.1), 1)
                test_indices = np.random.choice(
                    non_null_indices, size=test_size, replace=False
                )

                original_values = eval_df[col].iloc[test_indices].copy()
                temp_df = eval_df.copy()
                temp_df.loc[test_indices, col] = np.nan

                # Model Accuracy Evaluator Run
                if imputer is None or (
                    callable(imputer) and not hasattr(imputer, "fit")
                ):
                    if selected_model == "mean_imputation":
                        temp_imputed = temp_df.fillna(temp_df.mean())
                    else:
                        temp_imputed = temp_df.fillna(method="ffill").fillna(
                            method="bfill"
                        )
                elif callable(imputer) and not hasattr(imputer, "fit"):
                    temp_imputed = imputer(temp_df)
                else:
                    try:
                        if (
                            imputer.__class__.__name__ == "ARIMAImputationModel"
                            and model_params
                        ):
                            order = tuple(model_params.get("order", (1, 1, 1)))
                            seasonal_order = tuple(
                                model_params.get("seasonal_order", (0, 0, 0, 0))
                            )
                            y_column = model_params.get("y_column", None)
                            imputer = ARIMAImputationModel(
                                order=order,
                                seasonal_order=seasonal_order,
                                y_column=y_column,
                                use_forward_backward=True,
                            )
                            temp_imputed = imputer.transform(temp_df)
                        elif imputer.__class__.__name__ == "GradientBoostingImputationModel" and model_params:
                            imputer = GradientBoostingImputationModel(
                                {
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
                                    "yColumn": y_column,
                                }
                            )
                            imputer.fit(temp_df)
                            temp_imputed = imputer.impute(temp_df)
                        elif imputer.__class__.__name__ == "LSTMImputationModel":
                            if model_params:
                                imputer = LSTMImputationModel(
                                    params={
                                        "epochs": 10,
                                        "dropout_rate": 0.2,
                                        "learning_rate": 0.001,
                                        "yColumn": y_column,
                                        "use_forward_backward": True
                                    },
                                    df=temp_df,
                                    save_path=f"models/checkpoints"
                                )
                                imputer.fit()
                                temp_imputed = imputer.impute()
                        else:
                            imputer.fit(temp_df) # type: ignore
                            temp_imputed = imputer.transform(temp_df) # type: ignore
                    except:
                        temp_imputed = temp_df.fillna(method="ffill").fillna(
                            method="bfill"
                        )

                if isinstance(temp_imputed, pd.DataFrame) and len(temp_imputed) != len(
                    temp_df
                ):
                    if "index" in temp_imputed.columns:
                        labels = temp_imputed["index"]
                        vals   = temp_imputed[col].values
                    else:
                        labels = temp_imputed.index
                        vals   = temp_imputed[col].values

                    full_imputed = temp_df.copy()
                    full_imputed.loc[labels, col] = vals
                    temp_imputed = full_imputed

                imputed_values = temp_imputed[col].iloc[test_indices] # type: ignore
                valid = ~pd.isna(imputed_values)
                original_values = original_values[valid]
                imputed_values  = imputed_values[valid]

                rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
                mae = mean_absolute_error(original_values, imputed_values)

                eval_metrics[col] = {"rmse": float(rmse), "mae": float(mae)}

        if eval_metrics:
            avg_rmse = np.mean([m["rmse"] for m in eval_metrics.values()])
            avg_mae = np.mean([m["mae"] for m in eval_metrics.values()])
            evaluation_metrics = {
                "avg_rmse": float(avg_rmse),
                "avg_mae": float(avg_mae),
                "success": True if eval_metrics else False
            }
        else:
            evaluation_metrics = {
                "rmse": "Failed",
                "mae": "Failed",
                "success": False
            }
    except Exception as e:
        print(f"{datetime.now()} - Error during evaluation: {str(e)}")
        evaluation_metrics = {"rmse": "Failed", "mae": "Failed", "success": False}

    model_performance_records[dataset_id] = {
        **evaluation_metrics,
        "selected_model": selected_model,
    }
    save_model_performance_records()

    return {
        "dataset_id": dataset_id,
        "selected_model": selected_model,
        "imputed_data": (
            imputed_data if isinstance(imputed_data, dict) else imputed_data.to_dict()
        ),
        "evaluation_metrics": evaluation_metrics,
    }


@app.post("/imputed_dataset")
async def get_imputed_dataset(request: ImputeRequest):
    """
    Retrieve the imputed dataset for visualization with both original and imputed values.
    Similar to the /dataset endpoint but returns both original and imputed values for comparison.
    """
    dataset_id = request.dataset_id
    selected_model = request.selected_model
    selected_y_column = request.selected_y_column
    model_params = request.model_params

    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset_id not in model_performance_records:
        raise HTTPException(
            status_code=404,
            detail="This dataset has not been imputed yet. Please run imputation first.",
        )

    df = preprocessed_data[dataset_id]["data"].copy()

    selected_model = selected_model if selected_model else model_performance_records.get(dataset_id, {}).get("selected_model")
    if not selected_model:
        selected_model = "linear_interpolation"

    try:
        def get_model_by_name(model_name: str):
            model_mapping = {
                "linear_interpolation": LinearInterpolationModel(),
                "spline_interpolation": SplineImputationModel(),
                "mean_imputation": lambda df: df.fillna(df.mean()),
                "knn_imputation": KNNImputationModel(),
                "regression_imputation": RegressionImputationModel(),
                "mice_imputation": MICEImputationModel(),
                "arima_imputation": ARIMAImputationModel(),
                "gb_imputation": GradientBoostingImputationModel(),
                "lstm_imputation": None,
                "exponential_smoothing": ExponentialSmoothingModel()
            }
            return model_mapping.get(model_name.lower())

        # Get the imputation model
        imputer = get_model_by_name(selected_model)
        print(f"[GET IMPUTATION] Selected imputer: {imputer.__class__.__name__ if imputer else None}")

        # Impute using the same logic as in the /impute endpoint
        if imputer is None:
            if selected_model == "mean_imputation":
                imputed_df = df.fillna(df.mean())
            else:
                imputed_df = df.fillna(method="ffill").fillna(method="bfill") # type: ignore
        elif callable(imputer) and not hasattr(imputer, "fit"):
            imputed_df = imputer(df)
        else:
            try:
                if imputer.__class__.__name__ == "ARIMAImputationModel" and model_params:
                    order = tuple(model_params.get("order", (1, 1, 1)))
                    seasonal_order = tuple(model_params.get("seasonal_order", (1, 1, 1, 1)))
                    y_column = model_params.get("y_column", None)
                    hasSeasonality = model_params.get("hasSeasonality", False)
                    imputer = ARIMAImputationModel(
                        order=order,
                        seasonal_order=seasonal_order,
                        y_column=y_column,
                        use_forward_backward=True,
                        seasonality=hasSeasonality,
                    )
                    imputed_df = imputer.transform(df)
                elif imputer.__class__.__name__ == "GradientBoostingImputationModel":
                    y_column = selected_y_column if selected_y_column else None
                    imputer = GradientBoostingImputationModel(
                        {
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
                            "yColumn": y_column,
                        }
                    )
                    imputer.fit(df)
                    imputed_df = imputer.impute(df)
                elif imputer.__class__.__name__ == "LSTMImputationModel":
                    if model_params:
                        imputer = LSTMImputationModel(
                            params={
                                "epochs": 10,
                                "dropout_rate": 0.2,
                                "learning_rate": 0.001,
                                "yColumn": selected_y_column,
                                "use_forward_backward": True
                            },
                            df=df,
                            save_path=f"models/checkpoints"
                        )
                    imputer.fit()
                    imputed_df = imputer.impute()
                else:
                    imputer.fit(df.copy())
                    imputed_df = imputer.transform(df)
            except Exception as e:
                print(
                    f"{datetime.now()} - Error using {selected_model}: {str(e)}. Falling back to basic imputation."
                )
                imputed_df = df.fillna(method="ffill").fillna(method="bfill") # type: ignore
    except Exception as e:
        print(f"{datetime.now()} - Error during imputation retrieval: {str(e)}")
        imputed_df = df.fillna(method="ffill").fillna(method="bfill") # type: ignore

    y_column = selected_y_column if selected_y_column else model_params.get("y_column", None) if model_params else None

    if isinstance(imputed_df, dict):
        imputed_df = pd.DataFrame.from_dict(imputed_df)
    for frame in (df, imputed_df):
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if frame.index.name == "Date":
            frame.reset_index(drop=True, inplace=True)

    # Always 'Date' is the X column
    use_index_as_x = True
    x_is_datetime = False
    x_series = df["Date"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Date" in imputed_df.columns:
        imputed_df["Date"] = pd.to_datetime(imputed_df["Date"], errors="coerce")
    merged_df = pd.merge(
        df, imputed_df, on="Date", how="left", suffixes=("", "_imputed")
    )

    data_points = []
    numeric_index = range(len(df))

    for i, row in merged_df.iterrows():
        x_value = row['Date'].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row['Date']) else None
        is_missing = pd.isna(row[y_column]) if y_column is not None else True
        y_original = None if is_missing else extract_number(row[y_column]) if y_column is not None else None

        # Get imputed Y value
        y_imputed = None
        imputed_column = f"{y_column}_imputed"
        if imputed_column in row and pd.notna(row[imputed_column]):
            y_imputed = float(row[imputed_column])

        data_points.append(
            {
                "x": x_value,
                "y_original": y_original,
                "y_imputed": y_imputed,
                "missing": is_missing,
            }
        )

    return {
        "column_name": y_column,
        "x_column": "Date" if not use_index_as_x else "index",
        "data": data_points,
        "index_type": "datetime" if x_is_datetime else "numeric",
        "selected_model": selected_model,
    }


@app.get("/download_imputed/{dataset_id}")
async def download_imputed_dataset(dataset_id: str):
    """
    Download the imputed dataset as a CSV file.
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset_id not in model_performance_records:
        raise HTTPException(
            status_code=404,
            detail="This dataset has not been imputed yet. Please run imputation first.",
        )

    # Get the original dataset
    df = preprocessed_data[dataset_id]["data"].copy()

    # Retrieve selected model from performance records
    selected_model = model_performance_records.get(dataset_id, {}).get("selected_model")
    if not selected_model:
        # Use default imputation method if no model info
        selected_model = "linear_interpolation"

    # Import needed modules for imputation
    from models import (
        LinearInterpolationModel,
        SplineImputationModel,
        ExponentialSmoothingModel,
        ARIMAImputationModel,
        KNNImputationModel,
        RegressionImputationModel,
        MICEImputationModel,
        GradientBoostingImputationModel,
    )

    # Function to get model instance by name
    def get_model_by_name(model_name: str):
        model_mapping = {
            "linear_interpolation": LinearInterpolationModel(),
            "spline_interpolation": SplineImputationModel(),
            "mean_imputation": lambda df: df.fillna(df.mean()),
            "ffill": lambda df: df.fillna(method="ffill"),
            "bfill": lambda df: df.fillna(method="bfill"),
            "knn_imputation": KNNImputationModel(),
            "regression_imputation": RegressionImputationModel(),
            "mice_imputation": MICEImputationModel(),
            "arima_imputation": ARIMAImputationModel(),
            "gb_imputation": GradientBoostingImputationModel(),
            "lstm_imputation": None,  # Placeholder
            "exponential_smoothing": ExponentialSmoothingModel(),
        }
        return model_mapping.get(model_name.lower())

    # Get the imputation model and impute data
    imputer = get_model_by_name(selected_model)

    try:
        # Impute missing values using the same logic as in impute_dataset
        if imputer is None:
            if selected_model == "mean_imputation":
                imputed_df = df.fillna(df.mean())
            else:
                imputed_df = df.fillna(method="ffill").fillna(method="bfill")
        elif callable(imputer) and not hasattr(imputer, "fit"):
            imputed_df = imputer(df)
        else:
            try:
                imputer.fit(df.copy())
                imputed_df = imputer.transform(df)
            except Exception as e:
                print(
                    f"{datetime.now()} - Error using {selected_model}: {str(e)}. Falling back to basic imputation."
                )
                imputed_df = df.fillna(method="ffill").fillna(method="bfill")

        # Create a buffer to hold the CSV data
        buffer = io.StringIO()

        # If the index is a DatetimeIndex, include it as a column called 'Date'
        if isinstance(imputed_df.index, pd.DatetimeIndex):
            imputed_df = imputed_df.reset_index()

            # Rename the index column to 'Date' if it's not already named
            if imputed_df.columns[0] != "Date":
                imputed_df = imputed_df.rename(columns={imputed_df.columns[0]: "Date"})

        # Write the DataFrame to the buffer as CSV
        imputed_df.to_csv(buffer, index=True)

        # Set the pointer back to the beginning of the buffer
        buffer.seek(0)

        # Create filename based on original file and model used
        file_location = datasets[dataset_id]
        original_filename = os.path.basename(file_location)
        filename_without_ext = os.path.splitext(original_filename)[0]
        download_filename = f"{filename_without_ext}_imputed_{selected_model}.csv"

        # Return the buffer as a streaming response
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}"
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{datetime.now()} - Error processing download: {str(e)}",
        )


@app.get("/dataset_preview/{dataset_id}")
async def get_dataset_preview(dataset_id: str, rows: int = 5):
    """
    Get a preview of the dataset (first N rows) for display in a table.

    Args:
        dataset_id: The ID of the dataset to preview
        rows: Number of rows to return (default: 5)

    Returns:
        A JSON object with columns and data rows
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = preprocessed_data[dataset_id]["data"]

    preview_df = df.head(rows)

    columns = preview_df.columns.tolist()

    if "Date" in columns:
        columns.remove("Date")
        columns.insert(0, "Date")

    result = {"columns": columns, "data": []}

    # Convert rows to list of dictionaries
    for _, row in preview_df.iterrows():
        # Handle various data types for JSON serialization
        row_dict = {}
        for col in columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = None
            elif isinstance(value, (np.integer, int)):
                row_dict[col] = int(value)
            elif isinstance(value, (np.floating, float)):
                row_dict[col] = float(value)
            elif isinstance(value, (np.datetime64, pd.Timestamp)):
                # Convert numpy datetime64 to pandas Timestamp
                if isinstance(value, np.datetime64):
                    value = pd.Timestamp(value)
                row_dict[col] = value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                row_dict[col] = str(value)

        result["data"].append(row_dict)

    print(
        f"{datetime.now()} - Generated preview for dataset {dataset_id}: {len(result['data'])} rows, {len(result['columns'])} columns"
    )

    return result


@app.get("/arima_diagnostics/{dataset_id}")
async def get_arima_diagnostics(dataset_id: str, column: Optional[str] = None):
    """
    Generate diagnostic plots for ARIMA model analysis including:
    - Boxcox transformation plot
    - Differenced data plot
    - ACF plot
    - PACF plot

    These plots help with determining appropriate ARIMA parameters.

    Args:
        dataset_id: ID of the dataset to analyze
        column: The column to analyze (optional, will use first numeric column if not specified)

    Returns:
        Dictionary containing base64 encoded plot images
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get the dataset
    df = preprocessed_data[dataset_id]["data"].copy()

    # Import ARIMAImputationModel
    from models import ARIMAImputationModel

    # Create an instance of the model
    model = ARIMAImputationModel()

    try:
        plots = model.generate_diagnostic_plots(df, column)
        return plots
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating diagnostic plots: {str(e)}"
        )


@app.get("/gb_histogram/{dataset_id}")
async def get_gb_histogram(dataset_id: str, yColumn: Optional[str] = None):
    """
    Generate a histogram of the xgboost model target column values to find outliers.
    Args:
        dataset_id: ID of the dataset to analyze
        column: The column to analyze
    Returns:
        Dictionary containing base64 encoded plot images
    """
    if dataset_id not in preprocessed_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = preprocessed_data[dataset_id]["data"].copy()

    from models import GradientBoostingImputationModel
    model = GradientBoostingImputationModel()

    try:
        plots = model.generate_histogram(df, yColumn)
        return plots
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating histogram: {str(e)}"
        )


# ---------------------------
# Run using uvicorn
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
