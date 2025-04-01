# Time-Series Imputation Project

<p align="center">
  <img src="https://skillicons.dev/icons?i=git,python,fastapi,ai" /><br>
</p>
<br/>
<br/>

Implementation of ==time-series imputation system== that selects appropriate models based on dataset characteristics. The system uses a ==matrix-based lookup== approach combined with a ==decision tree== to determine the most suitable imputation method for different types of time-series data.

> Core Objectives
- Build a system for imputing missing values in time-series data
- Select appropriate models based on dataset characteristics
- Use a matrix-based lookup approach combined with a decision tree
- Evaluate performance against ground truth (last 10-20% of data)
- Provide an API interface with model selection override

> Key Considerations
- Different data types have different characteristics requiring different models
- Small datasets would not work with LSTM or transformers
- Performance metrics should be model-specific
- Preprocessing should be purely based on the dataset
- The system should handle all types of datasets

---

> Model Selection Matrix

***Matrix Structure***
- *Rows: Dataset characteristics/features*
  - Data size (small, medium, large)
  - Missing data percentage
  - Seasonality (none, weak, strong)
  - Trend (none, linear, non-linear)
  - Stationarity (stationary, non-stationary)
  - Complexity (low, medium, high)
  - Univariate vs. Multivariate
<br/>

- *Columns: Imputation models*
  - Simple models (mean, median, linear interpolation)
  - Intermediate models (spline, ARIMA, exponential smoothing)
  - Advanced models (KNN, regression-based, MICE)
  - LLM APis (gpt-4o, claude, ibm-granite)
<br/>

- *Cell Values: Performance metrics*
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R-squared

---

### Project Structure
```txt
imputation_app/
├── main.py                # FastAPI API entry point (as shown earlier)
├── preprocessing.py       # Preprocessing and feature extraction routines
├── models/                # Model implementations
│   ├── __init__.py
│   ├── linear_interpolation.py
│   ├── spline_interpolation.py
│   ├── exponential_smoothing.py
│   ├── arima_imputation.py
│   └── knn_imputation.py
│   └── regression_imputation.py
│   └── mice_imputation.py
│   └── gradient_boosting_imputation.py
│   └── custom_llm_imputation.py
├── selection.py           # Model selection logic (lookup matrix + decision tree)
├── evaluation.py          # Functions to split data and compute evaluation metrics
├── storage.py             # Persistence logic (e.g., file/database operations)
├── utils.py               # Utility functions (logging, helper functions)
└── requirements.txt       # List of dependencies
└── README.md              # project explanation & installation instructions
```
