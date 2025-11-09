<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-active-success.svg) ![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) ![Dataset](https://img.shields.io/badge/dataset-UCI%20ML-red.svg)

# Bike Rental Demand Prediction

Machine learning regression model for predicting daily bike rental demand using Capital Bikeshare data. Achieves 83.2% test R-squared and 81.5% cross-validation R-squared with Ridge regression.

</div>

---

## Overview

Predicts daily bike rental counts from temporal patterns (season, month, weekday) and weather conditions (temperature, humidity, windspeed). Addresses multicollinearity (VIF > 5), zero-inflated features, and non-normal distributions across 731 daily observations.

**Final Model:** Ridge Regression (alpha=0.464)

- Test: R-squared = 0.832, RMSE = 820 rentals
- Cross-validation: R-squared = 0.815 +/- 0.032
- Features: 26 (2 continuous + 24 categorical one-hot encoded)
- Overfitting gap: 0.5%

See [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) for why Ridge was selected over XGBoost despite 9% higher test error.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Load Trained Model

```python
import joblib
model = joblib.load('models/ridge_final.joblib')
scaler = joblib.load('models/scaler_final.joblib')

# Predict requires 26 features: atemp, windspeed_log + 24 categorical dummies
predictions = model.predict(X_new_scaled)
```

### Run Full Pipeline

Open `notebooks/PRCP-1018-BikeRental.ipynb` for complete data preparation, modeling, and evaluation workflow.

## Dataset

| Property | Details |
|----------|---------|
| **Source** | Capital Bikeshare System (Washington D.C., 2011-2012) |
| **Samples** | 731 daily records |
| **Features** | 16 attributes (15 predictors + count) |
| **Split** | 584 train / 147 test (80/20) |

See [docs/problem-statement.pdf](docs/problem-statement.pdf) for full metadata.

## Project Structure

**Core directories:**
- `data/raw/` - Original datasets (day.csv, hour.csv)
- `data/processed/` - Preprocessed train/test split
- `notebooks/` - Full analysis pipeline (PRCP-1018-BikeRental.ipynb)
- `src/` - Reusable modules (utils, statistical analysis, model evaluation)
- `models/` - Trained model artifacts (ridge_final.joblib, scaler_final.joblib)
- `reports/` - Detailed analysis reports
- `results/figures/` - Visualization outputs

## Working with the Notebook

**Import pattern used:**
The notebook imports functions from src/ modules using:
```python
from src.utils import memory_usage, dataframe_memory_usage, cap_outliers
from src.statistical_analysis import normality_test_with_skew_kurt, calculate_vif, spearman_correlation
from src.model_evaluation import evaluate_regression_model, hyperparameter_tuning, visualize_model_performance
```

**Running analysis:**
The notebook contains the full ML pipeline. Execute cells sequentially for:
1. Data loading and exploratory analysis
2. Feature engineering (log transform, one-hot encoding)
3. Multicollinearity resolution (VIF analysis, feature dropping)
4. Statistical analysis (normality tests, Spearman correlation)
5. Model comparison (10 algorithms tested)
6. Hyperparameter tuning (5-fold GridSearchCV)
7. Final model selection and persistence

## Model Training Workflow

**Base model evaluation:**
```python
from src.model_evaluation import evaluate_regression_model

metrics = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
# Returns: MAE, MSE, RMSE, R², Adjusted R², MSLE, MAPE, CV R², Training R², Overfit, Training Time
```

**Model visualization:**
```python
from src.model_evaluation import visualize_model_performance

visualize_model_performance(model, X_train, y_train, X_test, y_test)
# Generates 6-plot diagnostic grid: learning curve, true vs predicted, residuals, distributions, QQ plot
```

**Hyperparameter tuning:**
```python
from src.model_evaluation import hyperparameter_tuning

best_models, best_params, times = hyperparameter_tuning(
    models={'Ridge': Ridge()},
    param_grids={'Ridge': {'alpha': [0.001, 0.01, 0.1, 0.464, 1.0, 10.0, 100.0]}},
    X_train=X_train,
    y_train=y_train,
    scoring_metric='neg_mean_squared_error',
    cv_folds=5
)
```

## Statistical Analysis Functions

**Normality testing:**
```python
from src.statistical_analysis import normality_test_with_skew_kurt

normal_df, not_normal_df = normality_test_with_skew_kurt(df)
# Uses Shapiro-Wilk (n<=5000) or Kolmogorov-Smirnov (n>5000)
# Returns separate DataFrames for normal and non-normal columns with skewness/kurtosis
```

**Multicollinearity detection:**
```python
from src.statistical_analysis import calculate_vif

vif_data, high_vif_features = calculate_vif(data, exclude_target='cnt', multicollinearity_threshold=5.0)
# Returns VIF scores and features exceeding threshold
# In this project: dropped 'temp' (kept 'atemp'), VIF reduced from >5 to <3
```

**Spearman correlation:**
```python
from src.statistical_analysis import spearman_correlation_with_target

corr_data = spearman_correlation_with_target(
    data,
    non_normal_cols=['atemp', 'hum', 'windspeed_log'],
    target_col='cnt',
    plot=True,
    table=True
)
```

## Feature Engineering Pipeline

**Critical transformations applied:**

1. **Log transformation:** `windspeed` -> `windspeed_log` (handles zero-inflation)
2. **One-hot encoding:** season, month, weekday, weathersit (24 binary features)
3. **Multicollinearity resolution:** Dropped `temp` (VIF > 5, correlated with `atemp`)
4. **Dropped features:** `instant`, `dteday`, `casual`, `registered` (non-predictive or data leakage)
5. **Standardization:** StandardScaler on continuous features (`atemp`, `windspeed_log`)

**Final feature set (26 features):**
- Continuous (2): `atemp`, `windspeed_log`
- Categorical (24): season_*, mnth_*, weekday_*, weathersit_* (one-hot encoded)

## Model Persistence

**Loading the final model:**
```python
import joblib
import numpy as np

# Load artifacts
model = joblib.load('models/ridge_final.joblib')
scaler = joblib.load('models/scaler_final.joblib')

# Prepare features (example for single prediction)
features = np.array([[atemp, windspeed_log, season_Summer, season_Fall, ...]])  # 26 features
features[:, :2] = scaler.transform(features[:, :2])  # Scale continuous features only

# Predict
predictions = model.predict(features)
```

## Key Design Decisions

**Model selection criteria (weighted):**
1. Generalization (CV R² stability) - 40%
2. Accuracy (Test RMSE/R²) - 30%
3. Stability (overfitting gap) - 20%
4. Efficiency (training time, interpretability) - 10%

**Why Ridge over XGBoost:**
- XGBoost achieved lower test RMSE (748 vs 820) but showed 9.4% overfitting vs Ridge's 0.5%
- Ridge provides interpretability (26 transparent coefficients) vs XGBoost black box (200 trees)
- Training: 72x faster (0.002s vs 0.144s)
- Inference: 100x faster operations (26 multiplications vs tree traversal)
- CV consistency: Ridge 1.9-point CV-test gap vs XGBoost 2.5-point gap
- Trade-off: Accept 9% higher RMSE (72 rentals, 1.6% of mean demand) for 19x better generalization

**Feature engineering rationale:**
- **windspeed_log:** Original windspeed has 3% zero values; log transformation normalizes distribution
- **Dropped temp:** High correlation with atemp (r > 0.99); atemp is better predictor (feels-like temperature)
- **Keep all categoricals:** Season, month, weekday, weather capture temporal and environmental patterns

## Business Insights

**Top predictive features (Ridge coefficients):**
- Winter season: +1,676 rentals
- September: +894 rentals
- Apparent temperature (atemp): +758 rentals per unit increase
- Rain/snow weather: -2,149 rentals
- Year 2012: +1,528 rentals (45% growth from 2011)

**Operational recommendations:**
- Fleet rebalancing: Increase bikes in winter and September by 30-40%
- Weather-responsive staffing: Reduce operations 40-50% on rain/snow days
- Hourly model updates: Ridge's 0.002s training enables frequent retraining
- Prediction intervals: 95% CI = ±1,604 rentals for 4,504 average daily demand

## Reports

Detailed analysis in `reports/`:

- [Complete_Data_Analysis_Report.md](reports/Complete_Data_Analysis_Report.md) - Full methodology, statistical analysis, and results
- [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) - Model selection rationale and performance comparison
- [Challenges_Report.md](reports/Challenges_Report.md) - Technical challenges and solutions
- [GALLERY.md](results/figures/GALLERY.md) - Visualizations

## Development

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 src/

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Pre-commit Hooks

- black (88-char lines)
- isort (black-compatible)
- nbqa-black (notebooks)
- Validation (YAML, JSON, TOML, trailing whitespace, large files)

### Testing Configuration

**Flake8 settings:**
- Max line length: 88
- Ignored: E203, W503, E501
- Max complexity: 10
- Docstring convention: numpy

---

- MIT License - Copyright (c) 2025 Dhanesh B. B.
- GitHub: [https://github.com/dhaneshbb](https://github.com/dhaneshbb)
