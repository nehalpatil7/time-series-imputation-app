# models/custom_genai_imputation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from secrets.env
load_dotenv('secrets.env')

class CustomGenAIImputationModel:

    def __init__(self, model: str = "anthropic/claude-3.7-sonnet"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in secrets.env")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def fit(self, df: pd.DataFrame):
        # No local fitting; this model uses an external API.
        pass

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        # For illustration, summarize the dataset and call the genAI API.
        # Prepare a prompt that includes basic statistics.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        col = numeric_cols[0]
        missing_indices = df[df[col].isna()].index
        if missing_indices.empty:
            return df

        # Convert timestamps to simpler format for the API
        missing_dates = [str(idx).split()[0] for idx in missing_indices]

        summary = df.describe().to_string()
        prompt = (
            f"I have a time-series dataset with the following summary statistics:\n{summary}\n"
            f"The missing values are at these dates: {missing_dates}\n"
            "Please provide imputed values for these missing dates in the following JSON format:\n"
            '{"date_format_in_original_data": value}\n'
            "Use the dates exactly as shown above and provide numeric values only. "
            "Keep the response concise and complete."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an accurate time-series data imputation assistant. Always respond with valid JSON containing date-value pairs. Keep responses concise and complete.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,  # Increased max tokens
            )

            output_text = response.choices[0].message.content.strip()
            if not output_text:
                raise ValueError("Empty response from API")

            # Clean the response to ensure it's valid JSON
            output_text = output_text.replace("```json", "").replace("```", "").strip()

            # If the response appears truncated, try to complete it
            if output_text.count('{') > output_text.count('}'):
                output_text += '}'

            # Parse the JSON response
            import json
            try:
                imputed_mapping = json.loads(output_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {output_text}")
                # Fallback to forward/backward fill
                return df.ffill().bfill()

            imputed_df = df.copy()
            # Replace missing values using the mapping
            for date_str, value in imputed_mapping.items():
                try:
                    # Convert date string back to timestamp
                    idx = pd.Timestamp(date_str)
                    if idx in imputed_df.index and pd.isnull(imputed_df.loc[idx, col]):
                        imputed_df.loc[idx, col] = float(value)
                except Exception as e:
                    print(f"Error processing date {date_str}: {e}")
                    continue

            return imputed_df

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
