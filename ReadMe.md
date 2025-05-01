# Time-Series Imputation Project

<p align="center">
  <img src="https://skillicons.dev/icons?i=git,python,fastapi,ai" /><br>
</p>
<br/>
<br/>

Implementation of <mark>time-series imputation system</mark> that selects appropriate models based on dataset characteristics. The system uses a <mark>matrix-based lookup</mark> approach combined with a <mark>decision tree</mark> to determine the most suitable imputation method for different types of time-series data.

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

> Read Wiki to know more {https://github.com/nehalpatil7/time-series-imputation-app/wiki}
