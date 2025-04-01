from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import uuid
import os
import pandas as pd

from selection import extract_features, ModelSelector


app = FastAPI(title="Time Series Imputation API", version="0.1")

# In-memory storage for simplicity (can be replace with database/file storage)
datasets = {}  # dataset_id -> file path or data object
preprocessed_data = {}  # dataset_id -> {"data": ..., "features": ...}
model_performance_records = {}  # dataset_id -> evaluation metrics
model_selector = ModelSelector.load()


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


class UpdateFeedbackRequest(BaseModel):
    dataset_id: str
    model_performance: Dict[str, Any]
    user_feedback: Optional[str] = None


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def test_app():
    return {"Congratulations": "Successfully started Regeneration APIs!"}


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

    # Load dataset using pandas (assumes CSV for simplicity)
    try:
        df = pd.read_csv(
            datasets[dataset_id], parse_dates=True, infer_datetime_format=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e}")

    # Simple preprocessing: if there is a 'date' column, set it as index.
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        except Exception as e:
            print(f"Warning: Error processing date column: {e}")

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
    dataset_id = request.dataset_id
    print(preprocessed_data)
    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Preprocessed data not found for this dataset"
        )

    features = preprocessed_data[dataset_id]["features"]

    # If the user provided an override, return that
    if request.override_model:
        selected_model = request.override_model
    else:
        # Use the model_selector to predict the best candidate based on features
        selected_model = model_selector.predict(features)

    return {"dataset_id": dataset_id, "selected_model": selected_model}


@app.post("/impute")
async def impute_dataset(request: ImputeRequest):
    """
    Impute the missing values in the dataset using the selected model.
    (Placeholder logic: replace with calls to your imputation model wrappers.)
    """
    dataset_id = request.dataset_id
    selected_model = request.selected_model
    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Preprocessed data not found for this dataset"
        )

    # Dummy imputation: here you would call the specific imputation model's .impute() method.
    # For now, we return the preprocessed data without change.
    imputed_data = (
        preprocessed_data[dataset_id]["data"]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Dummy evaluation metrics (replace with real evaluation logic)
    evaluation_metrics = {"rmse": 0.5, "mae": 0.3}
    model_performance_records[dataset_id] = evaluation_metrics

    return {
        "dataset_id": dataset_id,
        "selected_model": selected_model,
        "imputed_data": imputed_data.head(10).to_dict(),  # sending a sample for brevity
        "evaluation_metrics": evaluation_metrics,
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
    dataset_id = request.dataset_id
    # In a real implementation, update the model_selector with the performance data:
    # For demonstration, assume we consider the current dataset features and performance to indicate the best model.
    if dataset_id not in preprocessed_data:
        raise HTTPException(
            status_code=404, detail="Dataset not found for feedback update"
        )

    features = preprocessed_data[dataset_id]["features"]
    # Here you would use the performance data to decide the best model.
    # For now, we assume the best model reported in feedback is the best.
    best_model = request.model_performance.get(
        "best_model", model_selector.candidate_models[0]
    )
    model_selector.add_record(features, best_model)
    model_selector.train()
    # Save updated model_selector
    model_selector.save()

    return {"dataset_id": dataset_id, "status": "Feedback updated successfully"}


# ---------------------------
# Run the API using uvicorn if executed as a script.
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
