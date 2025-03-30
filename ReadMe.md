# Time-Series Imputation Project





### Proposed Project Structure

imputation_app/
├── main.py                # FastAPI API entry point (as shown earlier)
├── preprocessing.py       # Preprocessing and feature extraction routines
├── models.py              # Candidate imputation model wrappers
├── selection.py           # Model selection logic (lookup matrix + decision tree)
├── evaluation.py          # Functions to split data and compute evaluation metrics
├── storage.py             # Persistence logic (e.g., file/database operations)
├── utils.py               # Utility functions (logging, helper functions)
└── requirements.txt       # List of dependencies
