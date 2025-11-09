# Report on Challenges Faced: Bike Rental Demand Prediction

**Project:** Bike Rental Demand Prediction Using Capital Bikeshare Data (2011-2012)
**Dataset:** 731 daily records, 16 attributes (26 features after encoding)
**Report Date:** March 01, 2025
**Last Revised:** November 07, 2025

---

## Executive Summary

This report documents four challenges encountered during the bike rental demand prediction project: non-normal distributions (p < 0.0001), severe multicollinearity (VIF > 662), categorical encoding complexity (7 features to 24 encoded), and model selection trade-offs. Solutions included non-parametric methods, iterative VIF removal with Ridge regularization, drop-first encoding, and multi-criteria selection, achieving 83.2% test R2 with 0.5% overfitting.

**Key Outcomes:** All features analyzed with appropriate methods, multicollinearity controlled (VIF: 662 to 37), categorical features encoded without redundancy, generalization prioritized over test performance.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Non-Normal Feature Distributions](#1-non-normal-feature-distributions)
  - [1.1 Challenge](#11-challenge)
  - [1.2 Solution](#12-solution)
  - [1.3 Outcome](#13-outcome)
- [2. Severe Multicollinearity](#2-severe-multicollinearity)
  - [2.1 Challenge](#21-challenge)
  - [2.2 Solution](#22-solution)
  - [2.3 Outcome](#23-outcome)
- [3. Categorical Feature Encoding](#3-categorical-feature-encoding)
  - [3.1 Challenge](#31-challenge)
  - [3.2 Solution](#32-solution)
  - [3.3 Outcome](#33-outcome)
- [4. Model Selection and Overfitting](#4-model-selection-and-overfitting)
  - [4.1 Challenge](#41-challenge)
  - [4.2 Solution](#42-solution)
  - [4.3 Outcome](#43-outcome)
- [5. Integrated Summary](#5-integrated-summary)
- [6. Recommendations for Future Projects](#6-recommendations-for-future-projects)
  - [Data Analysis](#data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Documentation](#documentation)
- [7. Code Snippets](#7-code-snippets)
  - [Normality Testing](#normality-testing)
  - [VIF Removal](#vif-removal)
  - [One-Hot Encoding](#one-hot-encoding)
  - [Cross-Validation](#cross-validation)
- [Conclusion](#conclusion)

---

## 1. Non-Normal Feature Distributions

### 1.1 Challenge

All 6 numerical features rejected Shapiro-Wilk normality test (p < 0.05):

| Feature | Test Statistic | p-value | Skewness | Kurtosis | Distribution Type |
|---------|----------------|---------|----------|----------|-------------------|
| temp | 0.966 | 5.1e-12 | -0.054 | -1.119 | Platykurtic |
| atemp | 0.974 | 3.7e-10 | -0.131 | -0.987 | Platykurtic |
| hum | 0.990 | 7.5e-05 | 0.046 | -0.606 | Near-symmetric, light tails |
| windspeed | 0.970 | 5.4e-11 | 0.546 | -0.145 | Right-skewed |
| windspeed_log | 0.985 | 6.7e-07 | 0.349 | -0.301 | Reduced skew after log |
| cnt (target) | 0.980 | 2.1e-08 | -0.047 | -0.815 | Near-symmetric, heavy tails |

**Impact:** Pearson correlation underestimated relationships (r=0.627 vs Spearman rho=0.643, +2.5%). Temperature variables showed bimodal seasonal patterns, windspeed concentrated at lower values.

### 1.2 Solution

**Tier 1: Non-Parametric Methods**

| Parametric Method | Non-Parametric Alternative | Reason |
|-------------------|---------------------------|--------|
| Pearson correlation | Spearman correlation | Rank-based, no normality assumption |
| t-tests | Mann-Whitney U test | Compares medians |
| ANOVA | Kruskal-Wallis H test | Non-parametric group comparison |

**Tier 2: Targeted Transformations**

Log-transformed windspeed to reduce right skew:
- Before: Skewness = 0.546, p = 5.4e-11
- After: Skewness = 0.349 (36% reduction), p = 6.7e-07

```python
df['windspeed_log'] = np.log(df['windspeed'] + 0.001)
```

**Why Ridge Regression?** Resistant to non-normality in predictors, uses regularization for correlated features, residuals showed acceptable normality.

### 1.3 Outcome

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Correlation Detection | Pearson: 0.627 | Spearman: 0.643 | +2.5% |
| Multicollinearity ID | Missed atemp-temp | Detected rho=0.993 | Caught perfect collinearity |
| Windspeed Skewness | 0.546 | 0.349 | -36% |

Model: Test R2 = 0.832, residuals approximately normal, Durbin-Watson = 1.983.

---

## 2. Severe Multicollinearity

### 2.1 Challenge

**Extreme Correlations:**

| Feature Pair | Spearman rho | Impact |
|--------------|------------|--------|
| atemp ↔ temp | 0.993 | Nearly perfect correlation |
| windspeed_log ↔ windspeed | 0.980 | Log preserves relationship |

**VIF Analysis (Before):**

| Feature | VIF | Impact |
|---------|-----|--------|
| temp | 662.86 | Standard errors inflated 26x |
| atemp | 642.49 | Standard errors inflated 25x |
| windspeed_log | 98.22 | Standard errors inflated 10x |
| windspeed | 83.97 | Standard errors inflated 9x |
| holiday_1 / workingday_1 | inf | Perfect collinearity |

**Impact:** Coefficient instability, inflated standard errors (t-statistics unreliable), cannot separate effects, overfitting risk.

### 2.2 Solution

**Three-Stage Approach:**

**Stage 1: Feature Pair Selection**

| Feature Removed | Feature Retained | Rationale |
|-----------------|------------------|-----------|
| windspeed | windspeed_log | Better distribution (skewness: 0.546 to 0.349) |
| temp | atemp | Apparent temp better predictor of behavior |
| hum | (none) | VIF=27.34, weak correlation with target (rho=0.09) |
| workingday_1 | holiday_1 | Clearer business interpretation (-477 rentals) |

**Stage 2: Iterative VIF Removal**

```python
def calculate_vif(data, threshold=8.0):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i)
                       for i in range(data.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Iteratively remove highest VIF features
while vif_df['VIF'].max() >= 8.0:
    to_remove = vif_df.iloc[0]['Feature']
    X_train = X_train.drop(columns=[to_remove])
```

**Stage 3: Retained High VIF Features**

| Feature | Final VIF | Justification |
|---------|-----------|---------------|
| atemp | 37.09 | Only temperature measure, primary weather variable |
| season_Fall | 14.34 | Non-redundant seasonal pattern |
| season_Summer | 9.98 | Summer demand peak |
| season_Winter | 9.82 | Winter demand patterns |
| mnth_Jul | 9.48 | Monthly granularity beyond seasons |
| mnth_Aug | 8.82 | Monthly granularity beyond seasons |

Ridge regularization (alpha=0.464) mitigates remaining multicollinearity through coefficient shrinkage.

### 2.3 Outcome

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max VIF | 662.86 (temp) | 37.09 (atemp) | 94.4% reduction |
| Features with VIF > 100 | 4 | 0 | 100% elimination |
| Model Overfitting | Not tested | 0.5% | Minimal |
| CV R2 Stability | Not tested | SD = 0.032 | Consistent |

**Coefficient Stability (OLS post-removal):**

| Feature | Coefficient | Std Error | p-value | 95% CI |
|---------|-------------|-----------|---------|--------|
| atemp | +758 | 78 | < 0.001 | [574, 881] |
| season_Winter | +1,676 | 222 | < 0.001 | [1,285, 2,156] |
| windspeed_log | -111 | 35 | 0.002 | [-178, -41] |

Model Performance: Test R2 = 0.832, CV R2 = 0.815 +/- 0.032, Overfitting = 0.5%.

---

## 3. Categorical Feature Encoding

### 3.1 Challenge

| Feature | Categories | Encoded Features | Issue |
|---------|------------|------------------|-------|
| mnth | 12 | 11 (drop Jan) | Seasonal patterns with monthly granularity |
| weekday | 7 | 6 (drop Sunday) | Weekly cycles |
| season | 4 | 3 (drop Spring) | Baseline selection important |
| weathersit | 3 | 2 (drop Clear) | Weather impact interpretation |
| yr | 2 | 1 (drop 2011) | Year-over-year growth |
| holiday | 2 | 1 (drop 0) | Binary indicator |
| workingday | 2 | Dropped | Redundant with holiday |

**Total:** 24 encoded features from 7 categorical features.

**Problems:** Dummy variable trap (VIF = inf if all categories included), baseline selection ambiguity, feature explosion (731 samples / 26 features = 28 samples/feature), holiday-workingday redundancy.

### 3.2 Solution

**Four-Stage Strategy:**

**Stage 1: Interpretable Baselines**

| Feature | Baseline | Rationale |
|---------|----------|-----------|
| season | Spring | Lowest mean rentals (2,604/day), natural start |
| mnth | January | First month, conventional |
| weekday | Sunday | Lowest rentals (4,228/day) |
| weathersit | Clear | Best conditions |
| yr | 2011 | Earlier year for growth measurement |

**Stage 2: Drop-First Encoding**

```python
df_encoded = pd.get_dummies(df, columns=['season', 'mnth', 'weekday', 'weathersit', 'yr'],
                             drop_first=True, dtype=int)
```

**Stage 3: Remove Redundancy**

Dropped workingday_1 (rho approx -0.99 with holiday_1, VIF = inf). Retained holiday_1 for clearer business interpretation.

**Stage 4: Verification**

All feature groups encoded without dummy trap, coefficients interpretable relative to baselines.

### 3.3 Outcome

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Feature Count | 16 | 26 | Manageable |
| Samples/Feature Ratio | 45.7 | 28.1 | Acceptable (> 10) |
| Perfect Collinearity | 2 instances | 0 | Resolved |
| VIF = inf features | 2 | 0 | Eliminated |

**Top Coefficients:**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| yr_1 | +2,026 | 2012 has 2,026 more rentals than 2011 (45% growth) |
| season_Winter | +1,676 | Winter adds 1,676 over Spring |
| season_Summer | +953 | Summer adds 953 over Spring |
| mnth_Sep | +894 | September adds 894 over January |
| atemp | +758 | Per SD increase in temperature |
| weathersit_Light Snow & Rain | -2,149 | Snow/rain reduces by 2,149 from Clear |

Model: Test R2 = 0.832, all features significant (p < 0.05), F-statistic = 110.8 (p < 0.0001).

---

## 4. Model Selection and Overfitting

### 4.1 Challenge

**Accuracy vs Generalization Trade-off:**

| Model | Test RMSE | Test R2 | Training R2 | CV R2 | Overfit (Delta R2) | CV-Test Gap |
|-------|-----------|---------|-------------|-------|----------------|-------------|
| XGBoost (Tuned) | 748 | 0.860 | 0.955 | 0.835 | 0.094 | 0.025 |
| Gradient Boosting | 804 | 0.839 | 0.952 | 0.826 | 0.113 | 0.013 |
| Ridge (Tuned) | 820 | 0.832 | 0.838 | 0.815 | 0.005 | 0.017 |
| Lasso (Tuned) | 819 | 0.833 | 0.838 | 0.813 | 0.005 | 0.020 |
| Random Forest | 819 | 0.833 | 0.955 | 0.792 | 0.122 | 0.041 |

**XGBoost Issues:** Training R2 = 0.955 suggests memorization, 9.4% overfitting, 200 trees = uninterpretable.

**Ridge Advantages:** Training R2 = 0.838 (appropriate complexity), 0.5% overfitting, 26 interpretable coefficients.

**Trade-off Quantification:**
- RMSE difference: 72 rentals (1.6% of mean demand 4,504)
- Overfitting reduction: 19x better (0.5% vs 9.4%)
- Training speed: 72x faster (0.002s vs 0.144s)
- Interpretability: 26 coefficients vs 200 trees

### 4.2 Solution

**Multi-Criteria Decision Framework:**

| Criterion | Weight | Metric | Rationale |
|-----------|--------|--------|-----------|
| Generalization | 40% | CV R2 mean/SD | Most predictive of production performance |
| Accuracy | 30% | Test R2/RMSE | Important for business case |
| Stability | 20% | Delta R2 (train-test) | Reliability on new data |
| Efficiency | 10% | Training time, interpretability | Operational ease |

**Model Scoring:**

| Model | CV (40%) | Test (30%) | Stability (20%) | Efficiency (10%) | Total |
|-------|----------|------------|-----------------|------------------|-------|
| Ridge | 0.325 | 0.250 | 0.199 | 0.100 | 0.874 |
| Lasso | 0.325 | 0.250 | 0.199 | 0.086 | 0.860 |
| XGBoost | 0.334 | 0.258 | 0.181 | 0.014 | 0.787 |
| Gradient Boosting | 0.330 | 0.252 | 0.177 | 0.061 | 0.820 |

**Ridge vs Lasso:** Ridge selected for 5x faster training (0.002s vs 0.010s), retains all features (Lasso zeros 13), better for multicollinearity.

**Cross-Validation Detail:**

| Fold | Ridge R2 | XGBoost R2 | Winner |
|------|----------|------------|--------|
| 1 | 0.817 | 0.805 | Ridge (+1.2) |
| 2 | 0.757 | 0.792 | XGBoost (+3.5) |
| 3 | 0.853 | 0.862 | XGBoost (+0.9) |
| 4 | 0.818 | 0.845 | XGBoost (+2.7) |
| 5 | 0.831 | 0.871 | XGBoost (+4.0) |
| Mean | 0.815 | 0.835 | XGBoost (+2.0) |
| Std Dev | 0.032 | 0.036 | Ridge (more stable) |

Ridge shows lower variance and more consistent performance across folds.

### 4.3 Outcome

**Final Model: Ridge Regression (alpha=0.464)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test R2 | 0.832 | 83.2% variance explained |
| Test RMSE | 820 rentals | 18.2% of mean demand (4,504) |
| Test MAE | 595 rentals | 13.2% of mean |
| Training R2 | 0.838 | Appropriate complexity |
| Overfitting | 0.5% | Minimal gap |
| CV R2 | 0.815 +/- 0.032 | Stable across folds (SD = 3.2%) |
| Training Time | 0.002s | Allows hourly retraining |

**Cross-Validation Breakdown:**

| Fold | R2 | RMSE | Contains |
|------|-----|------|----------|
| 1 | 0.817 | 841 | Jan-Dec mix (balanced) |
| 2 | 0.757 | 986 | Difficult period (transitional weather) |
| 3 | 0.853 | 752 | Best fit (stable patterns) |
| 4 | 0.818 | 836 | Balanced |
| 5 | 0.831 | 806 | Above average |

**Coefficient Stability:**

| Feature | Mean Coefficient | Coefficient SD | CV (SD/Mean) |
|---------|------------------|----------------|--------------|
| yr_1 | +2,024 | 67 | 3.3% |
| weathersit_Light Snow & Rain | -2,151 | 203 | 9.4% |
| season_Winter | +1,679 | 228 | 13.6% |
| atemp | +761 | 81 | 10.6% |
| season_Summer | +951 | 225 | 23.7% |

All top features show CV < 25%, confirming stability.

---

## 5. Integrated Summary

| Challenge | Key Metric | Solution | Outcome |
|-----------|------------|----------|---------|
| Non-Normal Distributions | All 6 features p < 0.05 | Spearman correlation, log transform windspeed, Ridge | Windspeed skew: -36%, correlation +2.5%, residuals acceptable |
| Severe Multicollinearity | VIF = 662.86, rho = 0.993 | Iterative VIF removal, Ridge regularization (alpha=0.464) | Max VIF: 662 to 37 (94% reduction), overfitting: 0.5% |
| Categorical Encoding | 7 categorical to 24 encoded | Drop-first encoding, workingday removal, interpretable baselines | 0 dummy trap instances, all coefficients interpretable |
| Model Selection | XGBoost RMSE=748 vs Ridge=820 | Multi-criteria scoring, A/B testing plan | Ridge: 0.5% vs 9.4% overfitting, 72x faster training |

**Final Model Specifications:**

| Attribute | Value |
|-----------|-------|
| Algorithm | Ridge Regression (L2 penalty) |
| Hyperparameter | alpha = 0.464 |
| Features | 26 (2 continuous, 24 categorical) |
| Samples | 731 (584 train / 147 test, 80/20) |
| Performance | Test R2 = 0.832, RMSE = 820, CV R2 = 0.815 +/- 0.032 |
| Generalization | Overfitting = 0.5%, Training R2 = 0.838 |
| Efficiency | Training time = 0.002s (72x faster than XGBoost) |
| Interpretability | 26 linear coefficients (full transparency) |

---

## 6. Recommendations for Future Projects

### Data Analysis
1. Test normality early with Shapiro-Wilk, use non-parametric methods when p < 0.05
2. Visualize distributions with Q-Q plots, histograms, KDE to understand shape
3. Apply domain-driven transformations (log for right-skewed), preserve interpretability

### Feature Engineering
4. VIF Protocol: Calculate after encoding, set threshold (8-10), iteratively remove highest with domain justification, retain important features < 40 if regularized
5. Categorical Encoding: Always drop-first, select interpretable baselines, check redundancy with correlation/VIF
6. Use Ridge (L2) for multicollinearity, Lasso (L1) for feature selection

### Model Selection
7. Cross-validation mandatory: 5-fold minimum, investigate CV-test gaps > 3 points
8. Multi-criteria framework: Define weights before results (40% generalization, 30% accuracy, 20% stability, 10% efficiency)
9. Overfitting thresholds: < 1% minimal, 1-5% acceptable, 5-10% moderate, > 10% severe
10. A/B testing: Deploy safe model (70%), challenger (30%), monitor 60+ days

### Documentation
11. Document challenges, solutions, quantified outcomes
12. Explain trade-offs with objective metrics
13. Record random seeds, split ratios, hyperparameters, preprocessing steps
14. Translate technical metrics to business impact

---

## 7. Code Snippets

### Normality Testing

```python
from scipy.stats import shapiro
import pandas as pd

def test_normality(df, numerical_cols, alpha=0.05):
    results = []
    for col in numerical_cols:
        data = df[col].dropna()
        stat, p_value = shapiro(data)
        results.append({
            'Feature': col,
            'Statistic': stat,
            'p-value': p_value,
            'Normal': 'Yes' if p_value > alpha else 'No',
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis()
        })
    return pd.DataFrame(results).sort_values('p-value')
```

### VIF Removal

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X, threshold=8.0):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def iterative_vif_removal(X, threshold=8.0, max_iter=20):
    X_reduced = X.copy()
    for iteration in range(max_iter):
        vif_data = calculate_vif(X_reduced, threshold)
        if vif_data['VIF'].max() <= threshold:
            break
        to_remove = vif_data.iloc[0]['Feature']
        X_reduced = X_reduced.drop(columns=[to_remove])
    return X_reduced, vif_data
```

### One-Hot Encoding

```python
def encode_categoricals(df, cat_cols, drop_first=True):
    label_maps = {
        'mnth': {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'},
        'season': {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'},
        'weekday': {0:'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday',
                    4:'Thursday', 5:'Friday', 6:'Saturday'},
        'weathersit': {1:'Clear', 2:'Mist & Cloudy', 3:'Light Snow & Rain'}
    }
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first, dtype=int)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

def evaluate_model_cv(model, X, y, cv=5, random_state=42):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

    print(f"Cross-Validation Results ({cv}-fold):")
    print(f"  Mean R2: {cv_scores.mean():.3f}")
    print(f"  Std Dev: {cv_scores.std():.3f}")
    print(f"  Range: [{cv_scores.min():.3f}, {cv_scores.max():.3f}]")

    model.fit(X, y)
    train_r2 = model.score(X, y)
    print(f"Training R2: {train_r2:.3f}")
    print(f"Overfitting: {(train_r2 - cv_scores.mean()):.3f}")
    return cv_scores
```

---

## Conclusion

This project addressed four challenges through evidence-based solutions, prioritizing generalization, interpretability, and operational viability. Each decision involved quantified trade-offs: using non-parametric methods for non-normal data, removing multicollinear features despite information loss, encoding 24 categorical features with careful baseline selection, and choosing Ridge over XGBoost despite 9% higher test error.

The resulting Ridge model achieves 83.2% test R2 with 81.5% CV R2 (+/- 3.2%), 0.5% overfitting, 0.002-second training, and full interpretability through 26 linear coefficients. This balance positions the model for reliable production deployment.

**Key Takeaway:** Operational ML projects require problem-solving where statistical rigor, domain expertise, and business pragmatism converge. Small accuracy reductions (1-2% RMSE) are justified for large gains in generalization (19x lower overfitting), transparency (26 coefficients vs 200 trees), and efficiency (72x faster training).

**Production Deployment Confidence:**
1. Cross-validation: 5-fold CV R2 = 0.815 +/- 0.032 (stable)
2. Overfitting control: 0.5% gap (minimal memorization)
3. Interpretability: All 26 coefficients align with domain knowledge
4. Efficiency: 0.002s training allows hourly retraining
5. A/B test plan: Ridge (70%) vs XGBoost (30%) for live validation

**Recommended Next Steps:**
1. Deploy Ridge with monitoring dashboard (RMSE, MAE, residuals)
2. Implement A/B testing (Ridge 70%, XGBoost 30%, 60-day evaluation)
3. Establish retraining triggers (RMSE > 1,000 for 3+ days, monthly)
4. Collect 2013-2014 data for temporal validation
5. Explore time series methods (ARIMA, Prophet) for day-to-day momentum
6. Integrate external data (events, transit disruptions) for outlier explanation

---

**Report Prepared By:** Dhanesh B. B.
**Contact:** [GitHub](https://github.com/dhaneshbb)
**License:** MIT
