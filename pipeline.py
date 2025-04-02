# pipeline.py

import pandas as pd
import numpy as np
from preprocessing import (
    load_dataset,
    preprocess_dataset,
)
from evaluation import split_data, compute_metrics
from selection import extract_features, ModelSelector
from storage import save_object, load_object

# Import candidate models from the models folder
from models import (
    LinearInterpolationModel,
    SplineImputationModel,
    ExponentialSmoothingModel,
    ARIMAImputationModel,
    KNNImputationModel,
    RegressionImputationModel,
    MICEImputationModel,
    GradientBoostingImputationModel,
    CustomGenAIImputationModel,
)


def get_model_by_name(model_name: str, **kwargs):
    """
    Given a model name (string), return an instance of the corresponding imputation model.
    """
    model_mapping = {
        "linear_interpolation": LinearInterpolationModel,
        "spline_imputation": SplineImputationModel,
        "exponential_smoothing": ExponentialSmoothingModel,
        "arima_imputation": ARIMAImputationModel,
        "knn_imputation": KNNImputationModel,
        "regression_imputation": RegressionImputationModel,
        "mice_imputation": MICEImputationModel,
        "gradient_boosting_imputation": GradientBoostingImputationModel,
        "custom_genai_imputation": CustomGenAIImputationModel,
    }
    model_cls = model_mapping.get(model_name.lower())
    if model_cls is None:
        raise ValueError(f"Model {model_name} is not recognized.")
    return model_cls(**kwargs)


def run_pipeline(
    file_path: str, holdout_percent: float = 0.15, override_model: str = None
):
    # Step 1: Load dataset
    print("Loading dataset...")
    df = load_dataset(file_path)

    # Step 2: Preprocess dataset (clean, set date index, extract features)
    print("Preprocessing dataset...")
    df_processed, features = preprocess_dataset(df)
    print("Extracted features:", features)

    # Step 3: Split data into training and holdout sets
    train_df, holdout_df = split_data(df_processed, holdout_percent=holdout_percent)
    print(
        f"Data split into {len(train_df)} training rows and {len(holdout_df)} holdout rows."
    )

    # Step 4: Use ModelSelector to recommend a candidate model unless overridden
    model_selector = ModelSelector.load()
    if override_model:
        best_model_name = override_model
    else:
        best_model_name = model_selector.predict(features)
    print("Recommended Model:", best_model_name)

    # Step 5: Instantiate the imputation model based on the recommended model name
    # For the custom genAI model, pass the API key (replace with your actual key)
    if best_model_name == "custom_genai_imputation":
        imputation_model = get_model_by_name(
            best_model_name, api_key="your_api_key_here"
        )
    else:
        imputation_model = get_model_by_name(best_model_name)

    # Step 6: Fit the model on the training data
    print("Fitting the knn_imputation model on training data...")
    imputation_model.fit(train_df)

    # Step 7: Impute missing values on the holdout set
    print("Imputing missing values on holdout data...")
    imputed_holdout = imputation_model.transform(holdout_df)

    # Step 8: Evaluate the imputation performance on the holdout set
    metrics = imputation_model.evaluate(df_processed, holdout_df)
    print("Evaluation Metrics:", metrics)

    # Step 9: Update the ModelSelector with these features and the (assumed) best model,
    # then retrain and save it.
    model_selector.add_record(features, best_model_name)
    model_selector.train()
    model_selector.save()

    return metrics


if __name__ == "__main__":
    # For testing the pipeline, provide the path to a sample CSV dataset.
    # Ensure the file exists at the specified location.
    file_path = (
        "datasets/nflx.csv"
    )
    metrics = run_pipeline(file_path, holdout_percent=0.15)
    print("Pipeline completed with metrics:", metrics)
