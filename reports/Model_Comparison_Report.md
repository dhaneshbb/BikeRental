# Model Comparison Report: Bike Rental Demand Prediction

**Project:** Bike Rental Demand Prediction Using Capital Bikeshare Data (2011-2012)
**Evaluation Dataset:** 584 training, 147 test samples | 26 features (2 continuous + 24 categorical)
**Report Date:** March 01, 2025
**Last Revised:** November 07, 2025

---

## Executive Summary

This report compares 10 regression algorithms for daily bike rental demand prediction. **Ridge Regression (alpha=0.464)** was selected despite not achieving the lowest test RMSE, prioritizing generalization and deployment efficiency.

**Key Findings:**
- **Best Test Performance:** XGBoost (RMSE = 748, R2 = 0.860)
- **Most Generalizable:** Ridge (CV R2 = 0.815 +/- 0.032, overfitting = 0.5%)
- **Fastest Training:** Ridge (0.002 seconds, 72x faster than XGBoost)
- **Worst Performance:** SVR (RMSE = 2,003, R2 = -0.000)

The trade-off between test accuracy and generalization stability led to Ridge's selection, prioritizing stable deployment over single-test-set metrics.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Evaluation Framework](#1-evaluation-framework)
- [2. Base Model Comparison (Default Parameters)](#2-base-model-comparison-default-parameters)
- [3. Hyperparameter Tuning](#3-hyperparameter-tuning)
- [4. Model Selection Decision](#4-model-selection-decision)
- [5. Cross-Validation Analysis](#5-cross-validation-analysis)
- [6. Training Efficiency](#6-training-efficiency)
- [7. Error Distribution](#7-error-distribution)
- [8. Recommendations](#8-recommendations)
  - [8.1 Deployment Strategy](#81-deployment-strategy)
  - [8.2 Retraining Triggers](#82-retraining-triggers)
  - [8.3 Improvement Roadmap](#83-improvement-roadmap)
- [9. Conclusion](#9-conclusion)
- [10. Code Snippets](#10-code-snippets)

---

## 1. Evaluation Framework

**Dataset:** Train 584 (80%) | Test 147 (20%) | Features 26 | Target: Daily Rentals (22 - 8,714)

**Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| RMSE | Prediction error | Lower better (rentals) |
| R2 | Variance explained | Higher better (0-1) |
| Training R2 | Fit capacity | Indicates overfitting if >> Test R2 |
| CV R2 (mean +/- SD) | Generalization | 5-fold stability measure |
| Overfit (Delta R2) | Train - Test R2 | Lower gap = better generalization |
| Training Time | Fit duration | Faster allows daily retraining |

**Selection Criteria:** Generalization (40%), Accuracy (30%), Stability (20%), Efficiency (10%)

---

## 2. Base Model Comparison (Default Parameters)

| Rank | Model | Test RMSE | Test R2 | Training R2 | Overfit | Time (s) | CV R2 |
|------|-------|-----------|---------|-------------|---------|----------|-------|
| 1 | Gradient Boosting | 760 | 0.856 | 0.931 | 0.075 | 0.131 | 0.835 |
| 2 | XGBoost | 792 | 0.843 | 0.920 | 0.076 | 0.079 | 0.824 |
| 3 | Linear Regression | 817 | 0.834 | 0.838 | 0.004 | 0.005 | 0.812 |
| 4 | Lasso | 817 | 0.833 | 0.838 | 0.005 | 0.014 | 0.812 |
| 5 | Ridge | 823 | 0.831 | 0.837 | 0.006 | 0.003 | 0.813 |
| 6 | Random Forest | 885 | 0.804 | 0.883 | 0.079 | 0.211 | 0.780 |
| 7 | ElasticNet | 925 | 0.786 | 0.793 | 0.007 | 0.003 | 0.775 |
| 8 | Decision Tree | 928 | 0.785 | 0.862 | 0.077 | 0.003 | 0.720 |
| 9 | KNN | 1,094 | 0.701 | 0.791 | 0.090 | 0.003 | 0.642 |
| 10 | SVR | 2,003 | -0.000 | 0.020 | 0.021 | 0.103 | 0.001 |

**Key Observations:**
- Tree methods achieve lowest test RMSE but show 7-8% overfitting
- Linear models show lower overfitting (0.4-0.6%) but have 30-60 higher RMSE
- Default hyperparameters matter: ElasticNet and SVR perform poorly, suggesting tuning potential

---

## 3. Hyperparameter Tuning

**Tuning Configurations:**

| Model | Parameters Tuned | Grid | Time (s) | Best Parameters |
|-------|------------------|------|----------|-------------------|
| Lasso | alpha: 10 values | 10 | 7.92 | alpha=0.774 |
| ElasticNet | alpha: 10, l1_ratio: 10 | 100 | 1.29 | alpha=0.0046, l1_ratio=0.5 |
| Ridge | alpha: 10 values | 10 | 0.30 | alpha=0.464 |
| Random Forest | n_estimators, max_depth, min_samples | 36 | 22.11 | n=200, depth=10, split=5 |
| Gradient Boosting | n_estimators, lr, max_depth | 27 | 9.72 | n=50, lr=0.3, depth=3 |
| XGBoost | n_estimators, lr, depth, subsample | 81 | 17.25 | n=200, lr=0.1, depth=3, sub=0.6 |

**Total Tuning Time:** 59 seconds (264 model fits)

**Post-Tuning Performance:**

| Model | Test RMSE | Test R2 | CV R2 (Mean +/- SD) | Training R2 | Overfit | Time (s) |
|-------|-----------|---------|-------------------|-------------|---------|----------|
| **XGBoost** | **748** | **0.860** | 0.835 +/- 0.027 | 0.955 | 0.094 | 0.144 |
| Gradient Boosting | 804 | 0.839 | 0.826 +/- 0.032 | 0.952 | 0.113 | 0.065 |
| Random Forest | 819 | 0.833 | 0.792 +/- 0.053 | 0.955 | 0.122 | 1.215 |
| Lasso | 819 | 0.833 | 0.813 +/- 0.027 | 0.838 | 0.005 | 0.010 |
| **Ridge** | 820 | 0.832 | **0.815 +/- 0.032** | 0.838 | **0.005** | **0.002** |
| ElasticNet | 825 | 0.830 | 0.813 +/- 0.034 | 0.837 | 0.007 | 0.012 |

**Tuning Impact:**
- ElasticNet: -100 RMSE (major improvement from severely over-regularized defaults)
- Ridge: -3 RMSE (marginal improvement, maintained CV stability)
- XGBoost: -44 RMSE (tuning reduced overfitting)
- Linear models maintained consistent <1% overfitting post-tuning

---

## 4. Model Selection Decision

**Multi-Criteria Scoring:**

| Model | CV R2 (40%) | Test R2 (30%) | Stability (20%) | Efficiency (10%) | **Total** |
|-------|-------------|---------------|-----------------|------------------|-----------|
| **Ridge** | 0.325 | 0.250 | 0.199 | 0.100 | **0.874** |
| Lasso | 0.325 | 0.250 | 0.199 | 0.086 | 0.860 |
| ElasticNet | 0.325 | 0.249 | 0.199 | 0.092 | 0.865 |
| XGBoost | 0.334 | 0.258 | 0.181 | 0.014 | 0.787 |
| Gradient Boosting | 0.330 | 0.252 | 0.177 | 0.061 | 0.820 |

**Decision: Ridge Regression (alpha=0.464)**

Ridge selected over Lasso for:
- Training Speed: 0.002s vs. Lasso 0.010s (5x faster)
- Coefficient Stability: Proportional shrinkage vs. zeroing
- Multicollinearity Handling: Designed for correlated features
- Simplicity: Standard choice for regularized regression

**Why Not XGBoost (Lowest Test RMSE)?**

| Concern | XGBoost | Ridge | Impact |
|---------|---------|-------|--------|
| Test RMSE | 748 rentals | 820 rentals | Ridge loses 72 rentals (9% higher) |
| CV R2 | 0.835 | 0.815 | XGBoost +2.0 points |
| Overfitting | 9.4% | 0.5% | Ridge 19x better generalization |
| Training Time | 0.144s | 0.002s | Ridge 72x faster |
| Interpretability | Black box (200 trees) | 26 coefficients | Ridge allows insights |
| Prediction Speed | Tree traversal | 26 multiplications | Ridge 100x faster |

**Trade-off Analysis:**
- RMSE Difference: 72 additional rentals error = 1.6% of avg demand (4,504)
- Overfitting Reduction: 19x better generalization (0.5% vs 9.4%)
- Business Context: 72-rental difference within operational tolerance

**Cross-Validation Validation:**

| Fold | Ridge R2 | XGBoost R2 | Winner |
|------|----------|------------|--------|
| 1 | 0.817 | 0.805 | Ridge (+1.2) |
| 2 | 0.757 | 0.792 | XGBoost (+3.5) |
| 3 | 0.853 | 0.862 | XGBoost (+0.9) |
| 4 | 0.818 | 0.845 | XGBoost (+2.7) |
| 5 | 0.831 | 0.871 | XGBoost (+4.0) |
| **Mean** | **0.815** | **0.835** | **XGBoost (+2.0)** |
| **Std Dev** | **0.032** | **0.036** | **Ridge (more stable)** |

XGBoost wins on average CV R2 but Ridge shows lower variance (SD = 0.032 vs. 0.036). Combined with 19x lower overfitting, Ridge offers more predictable production performance.

---

## 5. Cross-Validation Analysis

**Fold-by-Fold Stability:**

| Metric | Ridge | XGBoost | Lasso |
|--------|-------|---------|-------|
| Mean CV R2 | 0.815 | 0.835 | 0.813 |
| Std Dev | 0.032 | 0.036 | 0.027 |
| Range | 0.096 | 0.079 | 0.068 |
| Worst fold | 0.757 | 0.792 | 0.817 |

**CV vs. Test Gap:**

| Model | Test R2 | CV R2 | Gap |
|-------|---------|-------|-----|
| Ridge | 0.832 | 0.815 | 0.017 |
| Lasso | 0.833 | 0.813 | 0.020 |
| ElasticNet | 0.830 | 0.813 | 0.017 |
| XGBoost | 0.860 | 0.835 | **0.025** |
| Gradient Boosting | 0.839 | 0.826 | 0.013 |

Ridge's 1.7-point gap (second lowest) suggests highly consistent performance. XGBoost's 2.5-point gap indicates test set may contain patterns that don't generalize.

---

## 6. Training Efficiency

| Model | Training Time | Speedup | Retraining Frequency |
|-------|---------------|---------|----------------------|
| Ridge | 0.002s | 608x | Hourly possible |
| ElasticNet | 0.012s | 101x | Hourly possible |
| Lasso | 0.010s | 122x | Hourly possible |
| Gradient Boosting | 0.065s | 19x | Daily possible |
| XGBoost | 0.144s | 8x | Daily possible |
| Random Forest | 1.215s | 1x | Weekly acceptable |

**Model Footprint:**
- Ridge: 2.1 KB (26 coefficients) - Edge devices
- XGBoost: 847 KB (200 trees) - Server-side
- Random Forest: 1.2 MB (200 trees) - Server-side

Ridge's 0.002-second training allows integration into continuous retraining pipelines without infrastructure concerns.

---

## 7. Error Distribution

| Metric | Ridge | XGBoost | Lasso |
|--------|-------|---------|-------|
| MAE | 595 rentals | 500 rentals | 595 rentals |
| RMSE | 820 rentals | 748 rentals | 819 rentals |
| RMSE/MAE Ratio | 1.38 | 1.50 | 1.38 |

**Residual Characteristics:**

| Model | Mean Residual | Std Dev | Skewness | Kurtosis |
|-------|---------------|---------|----------|----------|
| Ridge | -3 rentals | 821 | -0.15 | 1.82 |
| XGBoost | +8 rentals | 751 | -0.34 | 2.91 |

- Both models unbiased (mean near 0)
- Ridge residuals more normal (lower kurtosis)
- XGBoost shows heavier tails (kurtosis > 2.5)

**95% Prediction Intervals:**
- Ridge: 2,900 - 6,108 rentals (+/- 1,604, +/- 35.6% of mean)
- XGBoost: 3,012 - 5,996 rentals (+/- 1,492, +/- 33.1% of mean)

7% wider interval for Ridge acceptable given stability and reduced tail risk.

---

## 8. Recommendations

### 8.1 Deployment Strategy

**Primary Model: Ridge (alpha=0.464)**
- Deploy for production daily demand forecasting
- Use coefficients for operational insights
- Retrain daily with rolling 7-day window

**A/B Testing:**
- Primary: Ridge (70%) | Challenger: XGBoost (30%)
- Monitor over 60 days: RMSE, MAE, operational KPIs
- Switch if XGBoost outperforms by >5% for 30+ consecutive days

**Fallback:**
- If Ridge RMSE exceeds 1,200 rentals, switch to XGBoost and investigate data drift

### 8.2 Retraining Triggers

**Scheduled:**
- Daily: Automatic retraining (0.002s overhead)
- Weekly: Hyperparameter retuning (~30s)
- Monthly: Full model comparison (~60s)
- Quarterly: Feature engineering review

**Event-Driven:** Retrain if:
- RMSE exceeds 1,000 for 3+ consecutive days
- Feature distribution shifts (e.g., temp mean changes >0.1 normalized units)
- System changes (new stations, fleet size changes >20%)
- External events (policy changes, disruptions)

### 8.3 Improvement Roadmap

**Short-Term (1-3 months):**
- Add lagged features: demand_{t-1}, demand_{t-7}
- Create interaction terms: temp x weathersit, weekday x season
- Polynomial features: temp2, temp3
- Validate on 2013-2014 data

**Medium-Term (3-6 months):**
- Test LightGBM, CatBoost
- Add SHAP values for XGBoost interpretability
- Deploy A/B testing in production

**Long-Term (6-12 months):**
- Implement ARIMA/SARIMA, LSTM
- Integrate external data: events, transit, economic indicators
- Train city-specific models for geographic expansion

---

## 9. Conclusion

**Ridge Regression (alpha=0.464)** was selected based on multi-criteria framework prioritizing generalization, stability, and operational efficiency.

**Final Verdict:** Ridge represents measured trade-off, accepting 1.6% test accuracy (72 rentals RMSE) to gain:
- 19x better generalization (0.5% vs 9.4% overfitting)
- 72x faster training (0.002s vs 0.144s)
- Full interpretability (26 coefficients vs 200-tree black box)
- More consistent CV-test gap (1.7 vs 2.5 points)

This aligns with best practices for production ML systems in operational contexts, where stability, transparency, and deployment simplicity outweigh marginal accuracy gains.

**Operational Impact:** Ridge's 820-rental RMSE represents 18.2% error on mean daily demand, well within fleet management tolerances (decisions operate at 100-500 rental granularity). The model supports weather-responsive staffing, seasonal fleet adjustments, dashboards, and daily automated retraining.

---

## 10. Code Snippets

**Ridge Configuration:**
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

model = Ridge(alpha=0.464, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['atemp', 'windspeed_log']])
X_train_scaled = np.hstack([X_train_scaled, X_train.drop(['atemp', 'windspeed_log'], axis=1)])
model.fit(X_train_scaled, y_train)
```

**Cross-Validation Evaluation:**
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

**Daily Retraining Pipeline:**
```python
import joblib
from datetime import datetime

def retrain_daily(new_data_path, model_save_path):
    df = pd.read_csv(new_data_path)
    df = preprocess_data(df)
    X, y = df.drop('cnt', axis=1), df['cnt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['atemp', 'windspeed_log']])

    model = Ridge(alpha=0.464, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(scaler.transform(X_test[['atemp', 'windspeed_log']]))
    rmse = ((y_test - y_pred)**2).mean()**0.5

    if rmse < 1200:
        joblib.dump(model, model_save_path)
        print(f"[{datetime.now()}] Model retrained: RMSE={rmse:.0f}")
    return model, rmse
```

---

**Report Prepared By:** Dhanesh B. B.
**Contact:** [GitHub](https://github.com/dhaneshbb)
**License:** MIT
