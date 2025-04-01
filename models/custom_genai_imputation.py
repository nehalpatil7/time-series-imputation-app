# models/custom_genai_imputation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import openai
from dotenv import load_dotenv
import os

# Load environment variables from secrets.env
load_dotenv('secrets.env')

class CustomGenAIImputationModel:
    def __init__(self, model: str = "claude-3.7-sonnet"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in secrets.env")
        openai.api_key = api_key
        self.model = model

    def fit(self, df: pd.DataFrame):
        # No local fitting; this model uses an external API.
        pass

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        # For illustration, summarize the dataset and call the genAI API.
        # Prepare a prompt that includes basic statistics.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary = df.describe().to_string()
        prompt = (
            f"I have a time-series dataset with the following summary statistics:\n{summary}\n"
            "Please suggest a complete imputation for missing values. "
            "Return a JSON object mapping the index (as string) to the imputed values for the first numeric column."
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful data imputation assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,  # Adjust based on your needs
            )
            # Extract response text
            output_text = response.choices[0].message.content
            if output_text is None:
                raise ValueError("Empty response from API")
            # Assume the output is a JSON mapping index to imputed value
            import json
            imputed_mapping = json.loads(output_text)

            imputed_df = df.copy()
            col = numeric_cols[0] if numeric_cols else None
            if col:
                # Replace missing values in the first numeric column using the mapping
                for idx, value in imputed_mapping.items():
                    # Ensure we update only if the original value is missing.
                    if pd.isnull(imputed_df.loc[idx, col]):
                        imputed_df.loc[idx, col] = value
                return imputed_df
            else:
                return df
        except Exception as e:
            print(f"Error calling genAI API: {e}")
            # Fallback: simply forward fill
            return df.ffill().bfill()

    def evaluate(self, df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
        imputed_holdout = self.impute(holdout)
        col = holdout.select_dtypes(include=[np.number]).columns[0]
        rmse = np.sqrt(mean_squared_error(holdout[col], imputed_holdout[col]))
        mae = mean_absolute_error(holdout[col], imputed_holdout[col])
        return {"rmse": rmse, "mae": mae}

model = CustomGenAIImputationModel()
