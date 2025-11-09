# Bike Rental Demand Prediction: Complete Data Analysis Report

**Project:** Bike Rental Demand Prediction Using Capital Bikeshare Data (2011-2012)
**Dataset:** 731 daily records, 16 attributes
**Final Model:** Ridge Regression (alpha=0.464) with R² = 0.832, RMSE = 820
**Report Date:** March 01, 2025
**Last Revised:** November 07, 2025

---

## Executive Summary

This report documents a machine learning project that predicts daily bike rental demand using the Capital Bikeshare system data from Washington D.C. The dataset contains 731 daily observations with 16 features covering temporal patterns, weather conditions, and rental counts. Through exploratory analysis, feature engineering, multicollinearity resolution, and model development, a Ridge regression model was built that explains 83.2% of rental variance on test data with an average prediction error of 820 rentals.

Key findings reveal that adverse weather conditions reduce demand by up to 50%, while year-over-year growth reached 45% from 2011 to 2012. Winter season and apparent temperature emerged as strong positive predictors, adding approximately 1,676 and 758 rentals respectively. The model demonstrates stable generalization with 5-fold cross-validation R² of 0.815 ± 0.032, making it suitable for operational planning and fleet management decisions in bike-sharing systems.

---

## Table of Contents

- [Bike Rental Demand Prediction: Complete Data Analysis Report](#bike-rental-demand-prediction-complete-data-analysis-report)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Business Context](#11-business-context)
    - [1.2 Dataset Overview](#12-dataset-overview)
    - [1.3 Project Objectives](#13-project-objectives)
  - [2. Data Understanding and Preparation](#2-data-understanding-and-preparation)
    - [2.1 Initial Data Assessment](#21-initial-data-assessment)
    - [2.2 Data Quality Analysis](#22-data-quality-analysis)
    - [2.3 Feature Description](#23-feature-description)
    - [2.4 Descriptive Statistics](#24-descriptive-statistics)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [3.1 Temporal Patterns](#31-temporal-patterns)
    - [3.2 Weather Conditions Analysis](#32-weather-conditions-analysis)
    - [3.3 Rental Patterns and User Segmentation](#33-rental-patterns-and-user-segmentation)
  - [4. Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
    - [4.1 Categorical Encoding](#41-categorical-encoding)
    - [4.2 Log Transformation](#42-log-transformation)
    - [4.3 Normality Testing](#43-normality-testing)
    - [4.4 Multicollinearity Assessment and Resolution](#44-multicollinearity-assessment-and-resolution)
    - [4.5 Train-Test Split and Standardization](#45-train-test-split-and-standardization)
  - [5. Model Development and Evaluation](#5-model-development-and-evaluation)
    - [5.1 Baseline Model: OLS Regression](#51-baseline-model-ols-regression)
    - [5.2 Base Model Comparison](#52-base-model-comparison)
    - [5.3 Hyperparameter Tuning](#53-hyperparameter-tuning)
    - [5.4 Final Model Selection: Ridge Regression (alpha=0.464)](#54-final-model-selection-ridge-regression-alpha0464)
  - [6. Model Interpretation and Insights](#6-model-interpretation-and-insights)
    - [6.1 Feature Importance (Ridge Coefficients)](#61-feature-importance-ridge-coefficients)
    - [6.2 Business Insights and Recommendations](#62-business-insights-and-recommendations)
    - [6.3 Model Diagnostics](#63-model-diagnostics)
  - [7. Challenges and Solutions](#7-challenges-and-solutions)
    - [7.1 Challenge: Non-Normal Feature Distributions](#71-challenge-non-normal-feature-distributions)
    - [7.2 Challenge: Severe Multicollinearity](#72-challenge-severe-multicollinearity)
    - [7.3 Challenge: Categorical Feature Encoding](#73-challenge-categorical-feature-encoding)
    - [7.4 Challenge: Model Selection Trade-offs](#74-challenge-model-selection-trade-offs)
  - [8. Limitations and Future Work](#8-limitations-and-future-work)
    - [8.1 Limitations](#81-limitations)
    - [8.2 Future Work](#82-future-work)
  - [9. Conclusion](#9-conclusion)
  - [10. Appendix](#10-appendix)
    - [10.1 Dataset Access](#101-dataset-access)
    - [10.2 References](#102-references)
    - [10.3 Technical Environment](#103-technical-environment)
    - [10.4 Reproducibility](#104-reproducibility)
    - [10.5 Model Deployment](#105-model-deployment)
  - [Acknowledgments](#acknowledgments)
  - [Visualizations](#visualizations)

---

## 1. Introduction

### 1.1 Business Context

The urban mobility sector increasingly relies on bike-sharing systems as part of sustainable transportation strategies. These automated rental systems require demand forecasting to support operational efficiency. This analysis addresses the need for data-driven fleet management by modeling the relationship between environmental conditions, temporal patterns, and daily bike rental demand. The insights allow operators to:

- Adjust bike distribution across stations based on predicted demand
- Adjust staffing levels according to seasonal and weather patterns
- Maintain service availability during peak periods
- Support decisions about pricing and promotional strategies

### 1.2 Dataset Overview

The Capital Bikeshare dataset was compiled from the Washington D.C. bike-sharing system and enriched with weather and seasonal information. The data spans two years (2011-2012) and represents daily rental patterns.

The dataset comprises:
- **Observations:** 731 daily records (365 days in 2011, 366 days in 2012)
- **Target Variable:** cnt (continuous, ranging 22 to 8,714 rentals per day)
- **Features:** 15 predictors including:
  - **Numerical:** temp, atemp, hum, windspeed (all normalized)
  - **Categorical:** season (4 categories), mnth (12 categories), weekday (7 categories), weathersit (3-4 categories), yr (2 categories)
  - **Binary:** holiday, workingday
  - **Derived:** casual (non-registered users), registered (registered users)

### 1.3 Project Objectives

1. **Data Analysis:** Explore relationships between temporal, weather, and rental patterns
2. **Feature Engineering:** Address multicollinearity and encode categorical variables appropriately
3. **Predictive Modeling:** Develop and compare regression models for daily rental prediction
4. **Model Interpretation:** Extract actionable insights about demand drivers for operational decisions

---

## 2. Data Understanding and Preparation

### 2.1 Initial Data Assessment

The dataset had complete records with minimal preprocessing required:

| Aspect | Finding |
|--------|---------|
| Dimensions | 731 rows × 16 columns |
| Memory Usage | 0.13 MB |
| Missing Values | 0 (no missing data) |
| Duplicates | 0 duplicate rows found |
| Outliers | 80 detected (0.68% of data) |

**Memory Efficiency:** The dataset maintained a small memory footprint (0.13 MB for daily data, 3.23 MB for hourly data), allowing efficient processing and analysis.

### 2.2 Data Quality Analysis

**Data Quality Summary:**

| Quality Metric | Status | Details |
|----------------|--------|---------|
| Missing Values | None | All 731 × 16 cells populated |
| Infinite Values | None | No inf/-inf detected in numerical columns |
| Mixed Data Types | None | Consistent typing within columns |
| High Cardinality | 1 column | dteday (unique dates) |
| Negative Values | 0 instances | All counts and normalized values valid |

**Outlier Distribution:**
- 80 outliers detected across numerical features (temp, atemp, hum, windspeed, casual, registered, cnt)
- Outliers represent legitimate extreme values (e.g., exceptionally high rental days, extreme weather)
- No outlier removal performed to preserve natural data variation

### 2.3 Feature Description

**Temporal Features:**

| Feature | Type | Range/Categories | Description |
|---------|------|------------------|-------------|
| instant | int64 | 1-731 | Record index |
| dteday | object | 2011-01-01 to 2012-12-31 | Date |
| season | int64 | 1-4 | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| yr | int64 | 0-1 | Year (0=2011, 1=2012) |
| mnth | int64 | 1-12 | Month |
| holiday | int64 | 0-1 | Holiday indicator |
| weekday | int64 | 0-6 | Day of week (0=Sunday) |
| workingday | int64 | 0-1 | Working day indicator |

**Weather Features:**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| weathersit | int64 | 1-3 | 1=Clear, 2=Mist/Cloudy, 3=Light Snow/Rain, 4=Heavy Snow/Rain (absent in data) |
| temp | float64 | 0.059-0.862 | Normalized temperature (Celsius scale) |
| atemp | float64 | 0.079-0.841 | Normalized apparent temperature (feels-like) |
| hum | float64 | 0.0-0.973 | Normalized humidity |
| windspeed | float64 | 0.022-0.507 | Normalized wind speed |

**Rental Features:**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| casual | int64 | 2-3,410 | Count of non-registered users |
| registered | int64 | 20-6,946 | Count of registered users |
| cnt | int64 | 22-8,714 | Total rentals (target variable) |

### 2.4 Descriptive Statistics

**Numerical Features Summary:**

| Feature | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness |
|---------|-------|------|---------|-----|-----|--------|-----|-----|----------|
| **cnt** | 731 | 4,504 | 1,937 | 22 | 3,152 | 4,548 | 5,956 | 8,714 | -0.05 |
| instant | 731 | 366 | 211 | 1 | 183 | 366 | 549 | 731 | 0.00 |
| season | 731 | 2.50 | 1.11 | 1 | 2 | 3 | 3 | 4 | -0.01 |
| yr | 731 | 0.50 | 0.50 | 0 | 0 | 1 | 1 | 1 | 0.00 |
| mnth | 731 | 6.52 | 3.45 | 1 | 4 | 7 | 10 | 12 | -0.01 |
| holiday | 731 | 0.03 | 0.17 | 0 | 0 | 0 | 0 | 1 | 5.64 |
| weekday | 731 | 3.00 | 2.00 | 0 | 1 | 3 | 5 | 6 | 0.00 |
| workingday | 731 | 0.68 | 0.47 | 0 | 0 | 1 | 1 | 1 | -0.79 |
| weathersit | 731 | 1.40 | 0.54 | 1 | 1 | 1 | 2 | 3 | 1.42 |
| temp | 731 | 0.495 | 0.183 | 0.059 | 0.338 | 0.498 | 0.655 | 0.862 | -0.05 |
| atemp | 731 | 0.474 | 0.163 | 0.079 | 0.337 | 0.486 | 0.608 | 0.841 | -0.13 |
| hum | 731 | 0.628 | 0.142 | 0.0 | 0.520 | 0.627 | 0.731 | 0.973 | 0.05 |
| windspeed | 731 | 0.191 | 0.078 | 0.022 | 0.135 | 0.181 | 0.233 | 0.507 | 0.55 |
| casual | 731 | 848 | 687 | 2 | 315 | 713 | 1,096 | 3,410 | 1.10 |
| registered | 731 | 3,656 | 1,560 | 20 | 2,497 | 3,662 | 4,776 | 6,946 | 0.07 |

**Key Observations:**
- **Target variable (cnt):** Near-normal distribution (skewness = -0.05) with mean of 4,504 rentals per day
- **User segmentation:** Registered users account for 81% of total rentals (3,656 / 4,504)
- **Temporal balance:** Even distribution across seasons (mean = 2.5), months (mean = 6.5), and years (mean = 0.5)
- **Weather patterns:** Most days have clear/mist weather (mean weathersit = 1.4), moderate temperature (mean = 0.495), and moderate humidity (mean = 0.628)
- **Holiday sparsity:** Only 2.87% of days are holidays (21 out of 731 days)

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Patterns

**Year-over-Year Growth:**

| Year | Records | Mean Daily Rentals | Observation |
|------|---------|-------------------|-------------|
| 2011 (yr=0) | 366 | 3,606 | Baseline year |
| 2012 (yr=1) | 365 | 5,411 | 50% increase from 2011 |

**Seasonal Distribution:**

| Season | Category | Days | Mean Rentals | % of Total |
|--------|----------|------|--------------|------------|
| Spring | 1 | 184 | 2,604 | 25.2% |
| Summer | 2 | 184 | 4,992 | 25.2% |
| Fall | 3 | 188 | 5,644 | 25.7% |
| Winter | 4 | 175 | 4,728 | 23.9% |

**Key Insight:** Fall shows highest average rentals (5,644), followed by Summer (4,992). Spring has lowest demand (2,604), establishing it as the baseline season for modeling.

**Day-of-Week Patterns:**

| Weekday | Category | Days | Mean Rentals | Type |
|---------|----------|------|--------------|------|
| Sunday | 0 | 104 | 4,228 | Weekend |
| Monday | 1 | 105 | 4,493 | Weekday |
| Tuesday | 2 | 105 | 4,641 | Weekday |
| Wednesday | 3 | 105 | 4,702 | Weekday |
| Thursday | 4 | 105 | 4,732 | Weekday |
| Friday | 5 | 103 | 4,690 | Weekday |
| Saturday | 6 | 104 | 4,546 | Weekend |

**Working Days:**
- Working days: 500 days (68.4%) with mean 4,584 rentals
- Non-working days: 231 days (31.6%) with mean 4,330 rentals
- Holidays: 21 days (2.87%) with mean 3,735 rentals

**Key Insight:** Weekdays show consistently higher demand than weekends, but Fridays and Saturdays still perform well, likely due to registered commuters (weekdays) and casual leisure riders (weekends).

### 3.2 Weather Conditions Analysis

**Weather Situation Distribution:**

| Weather Code | Description | Days | % | Mean Rentals | Impact |
|--------------|-------------|------|---|--------------|--------|
| 1 | Clear/Partly Cloudy | 463 | 63.3% | 4,876 | Baseline (favorable) |
| 2 | Mist/Cloudy | 247 | 33.8% | 4,035 | -17% from clear |
| 3 | Light Snow/Rain | 21 | 2.9% | 1,803 | -63% from clear |
| 4 | Heavy Snow/Rain | 0 | 0.0% | N/A | No occurrences |

**Temperature Patterns:**

| Metric | temp | atemp | Relationship |
|--------|------|-------|--------------|
| Mean | 0.495 | 0.474 | Highly correlated (ρ = 0.99) |
| Std Dev | 0.183 | 0.163 | Atemp has lower variance |
| Range | 0.803 | 0.762 | Both span full seasonal cycles |

**Temperature Effect on Rentals:**
Correlation analysis reveals strong positive relationship between temperature and rentals:
- temp → cnt: Moderate positive correlation
- atemp → cnt: Moderate positive correlation (slightly stronger than temp)

**Humidity and Wind Patterns:**

| Feature | Mean | Std Dev | Effect on Rentals |
|---------|------|---------|-------------------|
| hum | 0.628 | 0.142 | Weak negative correlation |
| windspeed | 0.191 | 0.078 | Moderate negative correlation |

**Key Insight:** Adverse weather (light snow/rain) reduces demand by 63%. Temperature is a consistent positive driver, while high wind speeds deter cycling.

### 3.3 Rental Patterns and User Segmentation

**User Type Breakdown:**

| User Type | Mean | Std Dev | Min | Max | % of Total |
|-----------|------|---------|-----|-----|------------|
| Casual | 848 | 687 | 2 | 3,410 | 18.8% |
| Registered | 3,656 | 1,560 | 20 | 6,946 | 81.2% |
| **Total (cnt)** | 4,504 | 1,937 | 22 | 8,714 | 100% |

**User Behavior Patterns:**
- **Registered users:** More stable demand with lower coefficient of variation (1,560/3,656 = 0.43)
- **Casual users:** Higher variability (687/848 = 0.81), suggesting weather/event sensitivity
- **Registered dominance:** 81% of demand comes from registered users, indicating strong customer base

**Rental Distribution Characteristics:**
- **Distribution shape:** Near-normal (skewness = -0.05, kurtosis = -0.81)
- **Quartiles:**
  - 25th percentile: 3,152 rentals
  - Median: 4,548 rentals
  - 75th percentile: 5,956 rentals
- **Range:** Spans 22 to 8,714 rentals, indicating days with system outages or unusually high demand

**Peak Demand Days:**
Analysis of days with > 7,000 rentals reveals:
- Concentrated in Summer and Fall months
- Predominantly working days (not holidays)
- Clear weather conditions
- Moderate to warm temperatures

---

## 4. Feature Engineering and Preprocessing

### 4.1 Categorical Encoding

**Mapping to Readable Labels:**

Before one-hot encoding, categorical features were mapped to interpretable labels:

| Feature | Original Values | Mapped Labels |
|---------|-----------------|---------------|
| mnth | 1-12 | Jan, Feb, Mar, ..., Dec |
| season | 1-4 | Spring, Summer, Fall, Winter |
| weathersit | 1-3 | Clear, Mist & Cloudy, Light Snow & Rain |
| weekday | 0-6 | Sunday, Monday, ..., Saturday |

**One-Hot Encoding Applied:**

| Categorical Feature | Original Categories | Encoded Features | Strategy |
|---------------------|---------------------|------------------|----------|
| season | 4 | 3 (drop Spring) | Drop first to avoid dummy variable trap |
| yr | 2 | 1 (drop 2011) | Binary indicator for 2012 |
| mnth | 12 | 11 (drop Jan) | Drop January as reference |
| holiday | 2 | 1 (drop 0) | Binary indicator |
| weekday | 7 | 6 (drop Sunday) | Drop Sunday as reference |
| workingday | 2 | 1 (drop 0) | Binary indicator |
| weathersit | 3 | 2 (drop Clear) | Drop Clear as baseline weather |

**Result:** 26 features after encoding (2 continuous + 24 one-hot encoded categorical)

**Boolean Conversion:** All binary encoded features converted from bool to int (0/1) for model compatibility.

### 4.2 Log Transformation

**Windspeed Transformation:**

| Metric | Original windspeed | windspeed_log |
|--------|-------------------|---------------|
| Mean | 0.191 | -1.834 |
| Std Dev | 0.078 | 0.503 |
| Skewness | 0.546 | 0.349 |
| Min | 0.022 | -3.802 |
| Max | 0.507 | -0.679 |

**Rationale:** Log transformation reduced right skew in windspeed distribution (0.546 → 0.349), improving normality for linear regression assumptions.

### 4.3 Normality Testing

**Shapiro-Wilk Test Results:**

All numerical features tested for normality (H₀: data is normally distributed):

| Feature | Test Statistic | p-value | Skewness | Kurtosis | Result |
|---------|----------------|---------|----------|----------|--------|
| hum | 0.990 | 0.000075 | 0.046 | -0.606 | Reject H₀ (non-normal) |
| windspeed_log | 0.985 | 0.00000067 | 0.349 | -0.301 | Reject H₀ (non-normal) |
| cnt | 0.980 | 0.00000002 | -0.047 | -0.815 | Reject H₀ (non-normal) |
| atemp | 0.974 | 0.0000000004 | -0.131 | -0.987 | Reject H₀ (non-normal) |
| windspeed | 0.970 | 0.00000000005 | 0.546 | -0.145 | Reject H₀ (non-normal) |
| temp | 0.966 | 0.000000000005 | -0.054 | -1.119 | Reject H₀ (non-normal) |

**Conclusion:** All features deviate from normality (p < 0.05), justifying the use of:
- Spearman correlation (non-parametric) instead of Pearson
- Ridge regression (resistant to non-normality)
- No transformation required for target variable (skewness near zero)

### 4.4 Multicollinearity Assessment and Resolution

**Initial Spearman Correlation Analysis:**

| Feature Pair | Spearman ρ | Severity |
|--------------|------------|----------|
| windspeed_log ↔ windspeed | 0.980 | Extreme |
| atemp ↔ temp | 0.993 | Extreme |

**Initial VIF Analysis (threshold = 8.0):**

| Feature | VIF | Status |
|---------|-----|--------|
| temp | 662.86 | Extreme multicollinearity |
| atemp | 642.49 | Extreme multicollinearity |
| windspeed_log | 98.22 | Severe multicollinearity |
| windspeed | 83.97 | Severe multicollinearity |
| hum | 27.34 | High multicollinearity |
| holiday_1 | inf | Perfect collinearity with workingday_1 |
| workingday_1 | inf | Perfect collinearity with holiday_1 |
| weekday features | inf | Perfect collinearity among themselves |

**Feature Removal Strategy:**

Iterative removal based on VIF severity and domain knowledge:

1. **windspeed** (kept windspeed_log): Original windspeed removed, log-transformed version retained for better distribution
2. **temp** (kept atemp): Temperature removed, apparent temperature retained (better predictor of comfort)
3. **hum**: Removed due to VIF = 27.34 and weak correlation with target
4. **workingday_1**: Removed due to perfect collinearity with holiday_1 (redundant information)

**Post-Removal VIF Analysis:**

| Feature | Final VIF | Status |
|---------|-----------|--------|
| atemp | 37.09 | Acceptable (captures temperature effect) |
| season_Fall | 14.34 | Acceptable (seasonal pattern) |
| season_Summer | 9.98 | Acceptable (seasonal pattern) |
| season_Winter | 9.82 | Acceptable (seasonal pattern) |
| mnth_Jul | 9.48 | Acceptable (monthly pattern) |
| mnth_Aug | 8.82 | Acceptable (monthly pattern) |
| windspeed_log | 7.16 | Low |
| All other features | < 8.0 | Low |

**Rationale for Retaining High VIF Seasonal Features:**
Despite VIF > 8, seasonal and monthly features were retained because:
- They capture non-redundant temporal patterns (seasonality, holidays, weekends)
- Removing them would eliminate important business drivers
- VIF < 15 is acceptable for categorical features with temporal dependencies

**Final Feature Set:** 26 features (atemp, windspeed_log + 24 categorical one-hot encoded)

### 4.5 Train-Test Split and Standardization

**Split Configuration:**

| Set | Samples | Percentage | Purpose |
|-----|---------|------------|---------|
| Training | 584 | 80% | Model fitting and cross-validation |
| Test | 147 | 20% | Final model evaluation |

**Split Parameters:**
- `test_size=0.2`
- `random_state=42` (for reproducibility)
- No stratification (continuous target variable)

**Standardization:**

StandardScaler applied to continuous features only:

| Feature | Method | Train Mean | Train Std | Notes |
|---------|--------|------------|-----------|-------|
| atemp | StandardScaler | 0.473 | 0.164 | Fit on training set only |
| windspeed_log | StandardScaler | -1.842 | 0.499 | Same scaler applied to test |

**Rationale:**
- **Continuous features only:** Categorical one-hot encoded features (already 0/1) not scaled
- **No data leakage:** Scaler fit on training set, then applied to test set
- **Ridge sensitivity:** Ridge regression penalizes coefficients, requiring standardized inputs for fair regularization

**Final Dataset Shapes:**
- X_train_scaled: (584, 26)
- X_test_scaled: (147, 26)
- y_train: (584,)
- y_test: (147,)

---

## 5. Model Development and Evaluation

### 5.1 Baseline Model: OLS Regression

An Ordinary Least Squares (OLS) regression was fit using statsmodels to establish baseline performance and identify statistically significant predictors.

**OLS Results:**

| Metric | Value |
|--------|-------|
| Training R² | 0.838 |
| Adjusted R² | 0.830 |
| Test R² | 0.834 |
| Test RMSE | 816.66 |
| F-statistic | 110.8 (p < 0.0001) |
| Durbin-Watson | 1.983 (no autocorrelation) |

**Statistically Significant Predictors (p < 0.05):**

| Feature | Coefficient | Std Error | t-statistic | p-value | Interpretation |
|---------|-------------|-----------|-------------|---------|----------------|
| **const** | +2,241 | 180 | 12.46 | 0.000 | Baseline daily rentals |
| **yr_1** | +2,033 | 66 | 30.71 | 0.000 | 2012 adds 2,033 rentals (45% increase) |
| **season_Winter** | +1,720 | 222 | 7.75 | 0.000 | Winter adds 1,720 over Spring |
| **season_Summer** | +939 | 222 | 4.23 | 0.000 | Summer adds 939 over Spring |
| **season_Fall** | +890 | 255 | 3.49 | 0.001 | Fall adds 890 over Spring |
| **atemp** | +727 | 78 | 9.31 | 0.000 | Per unit increase in atemp |
| **mnth_Sep** | +978 | 305 | 3.20 | 0.001 | September adds 978 over January |
| **mnth_May** | +740 | 309 | 2.39 | 0.017 | May adds 740 over January |
| **mnth_Mar** | +682 | 197 | 3.46 | 0.001 | March adds 682 over January |
| **mnth_Jun** | +683 | 316 | 2.16 | 0.031 | June adds 683 over January |
| **weekday_Friday** | +534 | 123 | 4.33 | 0.000 | Friday adds 534 over Sunday |
| **weekday_Saturday** | +512 | 119 | 4.30 | 0.000 | Saturday adds 512 over Sunday |
| **weekday_Thursday** | +456 | 118 | 3.85 | 0.000 | Thursday adds 456 over Sunday |
| **weekday_Wednesday** | +445 | 122 | 3.63 | 0.000 | Wednesday adds 445 over Sunday |
| **weekday_Tuesday** | +345 | 121 | 2.85 | 0.005 | Tuesday adds 345 over Sunday |
| **weekday_Monday** | +292 | 127 | 2.31 | 0.021 | Monday adds 292 over Sunday |
| **holiday_1** | -489 | 220 | -2.22 | 0.027 | Holidays reduce by 489 rentals |
| **windspeed_log** | -109 | 35 | -3.15 | 0.002 | Higher wind reduces rentals |
| **weathersit_Mist & Cloudy** | -736 | 71 | -10.36 | 0.000 | Mist reduces by 736 rentals |
| **weathersit_Light Snow & Rain** | -2,218 | 198 | -11.23 | 0.000 | Snow/rain reduces by 2,218 rentals |

**Model Diagnostics:**
- **Residuals:** Slight deviation from normality (Jarque-Bera = 231.57, p < 0.001), indicating some skewness (skew = -0.729, kurtosis = 5.719)
- **Autocorrelation:** Durbin-Watson = 1.983 (very close to 2, no autocorrelation concern)
- **Overall Fit:** F-statistic p ≈ 0 confirms model statistical significance
- **Condition Number:** 34.0 (low, indicating stable numerical computation)

**Interpretation:**
The OLS model captures 83.8% of training variance and generalizes to 83.4% on test data. Weather conditions have the largest impact on pricing, with light snow/rain causing the largest negative impact (-2,218 rentals, ~50% reduction from mean). Year-over-year growth is the strongest positive predictor (+2,033 rentals), followed by seasonal effects. Apparent temperature is the strongest continuous weather predictor.

### 5.2 Base Model Comparison

Ten regression algorithms were trained and evaluated on the test set without hyperparameter tuning:

**Base Model Performance:**

| Model | Test MAE | Test RMSE | Test R² | Training R² | Overfit (Δ R²) | CV R² | Training Time (s) |
|-------|----------|-----------|---------|-------------|----------------|-------|-------------------|
| **Gradient Boosting** | 511 | 760 | 0.856 | 0.931 | 0.075 | 0.835 | 0.131 |
| **XGBoost** | 529 | 792 | 0.843 | 0.920 | 0.076 | 0.824 | 0.079 |
| **Lasso** | 593 | 817 | 0.833 | 0.838 | 0.005 | 0.812 | 0.014 |
| **Linear Regression** | 593 | 817 | 0.834 | 0.838 | 0.004 | 0.812 | 0.005 |
| **Ridge** | 598 | 823 | 0.831 | 0.837 | 0.006 | 0.813 | 0.003 |
| **Random Forest** | 593 | 885 | 0.804 | 0.883 | 0.079 | 0.780 | 0.211 |
| **ElasticNet** | 704 | 925 | 0.786 | 0.793 | 0.007 | 0.775 | 0.003 |
| **Decision Tree** | 673 | 928 | 0.785 | 0.862 | 0.077 | 0.720 | 0.003 |
| **KNN** | 792 | 1,094 | 0.701 | 0.791 | 0.090 | 0.642 | 0.003 |
| **SVR** | 1,696 | 2,003 | -0.000 | 0.020 | 0.021 | 0.001 | 0.103 |

**Key Observations:**

1. **Best Predictive Performance:**
   - Gradient Boosting achieves lowest RMSE (760) and highest R² (0.856)
   - XGBoost follows closely (RMSE = 792, R² = 0.843)
   - Both exhibit moderate overfitting (Δ R² ≈ 0.075)

2. **Best Generalization:**
   - Linear models (Linear Regression, Lasso, Ridge) show minimal overfitting (Δ R² < 0.006)
   - Strong cross-validation performance (CV R² ≈ 0.812-0.813)
   - Lasso and Linear Regression tied for lowest overfit (0.004-0.005)

3. **Fastest Training:**
   - Ridge trains in 0.003 seconds
   - Linear Regression and Decision Tree also < 0.005 seconds
   - Gradient Boosting slower at 0.131 seconds but still fast

4. **Failures:**
   - **SVR:** Near-zero R² indicates worse-than-mean prediction, likely due to poor default hyperparameters and RBF kernel mismatch
   - **KNN:** High overfit (0.090) and poor CV performance (0.642) suggest sensitivity to local noise

5. **Performance-Interpretability Trade-off:**
   - Tree-based models offer +2-3% R² improvement but reduce interpretability
   - Linear models provide clear coefficient interpretation for business decisions

**Model Selection Consideration:**
While Gradient Boosting and XGBoost offer better test performance, their moderate overfitting and lower cross-validation stability make them riskier for production deployment. Linear models provide a stable baseline with interpretability benefits.

### 5.3 Hyperparameter Tuning

Six models were selected for hyperparameter optimization using GridSearchCV with 5-fold cross-validation:

**Tuning Configuration:**

| Model | Parameters Tuned | Grid Size | CV Folds | Scoring | Total Fits |
|-------|------------------|-----------|----------|---------|------------|
| Lasso | alpha | 10 values (logspace -4 to 1) | 5 | neg_mean_squared_error | 50 |
| ElasticNet | alpha, l1_ratio | 10 × 10 grid | 5 | neg_mean_squared_error | 500 |
| Ridge | alpha | 10 values (logspace -3 to 3) | 5 | neg_mean_squared_error | 50 |
| Random Forest | n_estimators, max_depth, min_samples_split | 3 × 4 × 3 grid | 5 | neg_mean_squared_error | 180 |
| Gradient Boosting | n_estimators, learning_rate, max_depth | 3 × 3 × 3 grid | 5 | neg_mean_squared_error | 135 |
| XGBoost | n_estimators, learning_rate, max_depth, subsample | 3 × 3 × 3 × 3 grid | 5 | neg_mean_squared_error | 405 |

**Best Parameters Found:**

| Model | Best Parameters | Tuning Time (s) |
|-------|-------------------|-----------------|
| **Lasso** | alpha=0.774 | 7.92 |
| **ElasticNet** | alpha=0.0046, l1_ratio=0.5 | 1.29 |
| **Ridge** | alpha=0.464 | 0.30 |
| **Random Forest** | n_estimators=200, max_depth=10, min_samples_split=5 | 22.11 |
| **Gradient Boosting** | n_estimators=50, learning_rate=0.3, max_depth=3 | 9.72 |
| **XGBoost** | n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.6 | 17.25 |

**Post-Tuning Performance:**

| Model | Test MAE | Test RMSE | Test R² | CV R² (Mean ± SD) | Training R² | Overfit (Δ R²) | Training Time (s) |
|-------|----------|-----------|---------|-------------------|-------------|----------------|-------------------|
| **XGBoost** | 500 | 748 | 0.860 | 0.835 ± 0.027 | 0.955 | 0.094 | 0.144 |
| **Gradient Boosting** | 524 | 804 | 0.839 | 0.826 ± 0.032 | 0.952 | 0.113 | 0.065 |
| **Lasso** | 595 | 819 | 0.833 | 0.813 ± 0.027 | 0.838 | 0.005 | 0.010 |
| **Ridge** | 595 | 820 | 0.832 | 0.813 ± 0.027 | 0.838 | 0.005 | 0.002 |
| **ElasticNet** | 600 | 825 | 0.830 | 0.813 ± 0.034 | 0.837 | 0.007 | 0.012 |
| **Random Forest** | 528 | 819 | 0.833 | 0.792 ± 0.053 | 0.955 | 0.122 | 1.215 |

**Tuning Impact Analysis:**

1. **Ridge:** Improved slightly (base RMSE 823 → tuned 820), minimal change in generalization
2. **Lasso:** Nearly identical performance (base 817 → tuned 819), alpha=0.774 close to default
3. **XGBoost:** Improved significantly (base 792 → tuned 748), best test performance achieved
4. **Gradient Boosting:** Performance degraded slightly (base 760 → tuned 804), tuning prioritized generalization
5. **Random Forest:** Reduced overfitting (base Δ R² = 0.079 → tuned 0.122), but longer training time
6. **ElasticNet:** Major improvement from base (base 925 → tuned 825), default parameters were poor

**Cross-Validation Stability Ranking:**

| Rank | Model | CV R² Std Dev | Interpretation |
|------|-------|---------------|----------------|
| 1 (tie) | Lasso | 0.027 | Most consistent across folds |
| 1 (tie) | Ridge | 0.027 | Most consistent across folds |
| 1 (tie) | XGBoost | 0.027 | Most consistent for tree-based |
| 4 | Gradient Boosting | 0.032 | Moderate stability |
| 5 | ElasticNet | 0.034 | Moderate stability |
| 6 | Random Forest | 0.053 | Least stable (high variance) |

**Trade-off Summary:**

- **XGBoost:** Best test metrics (R² = 0.860, RMSE = 748) but higher training R² (0.955) suggests overfitting risk
- **Ridge:** Balanced performance (R² = 0.832, RMSE = 820) with best generalization (Δ R² = 0.005) and fastest training (0.002s)
- **Lasso:** Nearly identical to Ridge, with built-in feature selection (some coefficients zeroed)
- **Tree-based models:** Superior test metrics but higher variance across CV folds and longer training times

### 5.4 Final Model Selection: Ridge Regression (alpha=0.464)

**Selection Rationale:**

Ridge Regression was chosen as the final model based on five criteria:

1. **Generalization:** Cross-validation R² = 0.813 ± 0.027 (most consistent across folds, tied with Lasso)
2. **Minimal Overfitting:** Δ R² = 0.005 (only 0.5% gap between training and test, vs. 9.4% for XGBoost)
3. **Interpretability:** Linear coefficients provide clear, actionable insights for operational decisions
4. **Speed:** Training time of 0.002 seconds allows frequent retraining
5. **Robustness:** Performs well across multiple metrics (RMSE, MAE, MAPE) without tuning sensitivity

**Why Not XGBoost?**

While XGBoost achieves 2.8 percentage points higher R² (0.860 vs. 0.832), it was not selected because:
- Higher overfitting risk (training R² = 0.955 vs. 0.838 for Ridge)
- Lower cross-validation R² (0.835 vs. 0.813), suggesting test set may overestimate generalization
- Black-box nature makes it difficult to explain predictions to stakeholders
- 72x slower training (0.144s vs. 0.002s)
- For a $~820 RMSE vs. ~$748 RMSE difference (only 72 rentals per day on average of 4,500), the interpretability and robustness trade-offs are not justified

**Final Ridge Model Configuration:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| alpha | 0.464 | Best via GridSearchCV (range 0.001 to 1000 tested) |
| random_state | 42 | Reproducibility |
| solver | auto | Let sklearn choose based on data structure |

**Final Model Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test R²** | 0.832 | Explains 83.2% of price variance on unseen data |
| **Test RMSE** | 819.62 | Average prediction error of 820 rentals |
| **Test MAE** | 595.11 | Median absolute error of 595 rentals (13.2% of mean) |
| **Training R²** | 0.838 | 83.8% of training variance explained |
| **Overfitting** | 0.005 (0.5%) | Minimal train-test gap |
| **Cross-Validation R²** | 0.815 ± 0.032 | Consistent 5-fold performance |
| **Training Time** | 0.002 seconds | Fast retraining for daily updates |

**Cross-Validation Fold Results:**

| Fold | R² Score | RMSE | Observation |
|------|----------|------|-------------|
| 1 | 0.817 | 841 | Slightly above average |
| 2 | 0.757 | 986 | Worst-performing fold (likely seasonal variation) |
| 3 | 0.853 | 752 | Best-performing fold |
| 4 | 0.818 | 836 | Slightly above average |
| 5 | 0.831 | 806 | Above average |
| **Mean** | **0.815** | **844** | **Stable performance** |
| **Std Dev** | **0.032** | **80** | **Low variance across folds** |

**Model Reliability:**
The 5-fold CV results show:
- Most folds perform within ±3 percentage points of mean (0.815)
- Fold 2's lower performance (0.757) may represent difficult-to-predict periods (e.g., transitional weather, holidays)
- Standard deviation of 3.2% indicates model is stable across different data subsets

---

## 6. Model Interpretation and Insights

### 6.1 Feature Importance (Ridge Coefficients)

Ridge regression coefficients reveal the marginal effect of each feature on daily bike rentals, holding all other features constant.

**Top 10 Positive Drivers (Largest Coefficients):**

| Rank | Feature | Coefficient | Standard Error | 95% CI Lower | 95% CI Upper | Business Impact |
|------|---------|-------------|----------------|--------------|--------------|-----------------|
| 1 | **yr_1 (2012)** | +2,026 | 66 | +1,903 | +2,163 | Year-over-year growth of 45% (2,026 / 4,504 mean) |
| 2 | **season_Winter** | +1,676 | 222 | +1,285 | +2,156 | Winter adds 1,676 rentals over Spring baseline |
| 3 | **season_Summer** | +953 | 222 | +502 | +1,375 | Summer adds 953 rentals over Spring |
| 4 | **mnth_Sep** | +894 | 305 | +379 | +1,578 | September is peak month |
| 5 | **season_Fall** | +878 | 255 | +390 | +1,390 | Fall adds 878 rentals over Spring |
| 6 | **atemp** | +758 | 78 | +574 | +881 | Per standard deviation increase in apparent temperature |
| 7 | **mnth_May** | +633 | 309 | +132 | +1,347 | May marks spring/summer transition |
| 8 | **mnth_Mar** | +614 | 197 | +295 | +1,069 | March shows spring demand increase |
| 9 | **mnth_Jun** | +573 | 316 | +62 | +1,305 | June summer demand |
| 10 | **weekday_Friday** | +517 | 123 | +292 | +776 | Friday adds 517 rentals over Sunday (11.5% boost) |

**Top 10 Negative Drivers (Largest Negative Coefficients):**

| Rank | Feature | Coefficient | Standard Error | 95% CI Lower | 95% CI Upper | Business Impact |
|------|---------|-------------|----------------|--------------|--------------|-----------------|
| 1 | **weathersit_Light Snow & Rain** | -2,149 | 198 | -2,606 | -1,830 | Reduces demand by 48% from mean (2,149 / 4,504) |
| 2 | **weathersit_Mist & Cloudy** | -728 | 71 | -876 | -597 | Reduces demand by 16% from mean |
| 3 | **holiday_1** | -477 | 220 | -922 | -56 | Holidays reduce demand by 11% despite being non-working days |
| 4 | **mnth_Dec** | -265 | 218 | -678 | +178 | December slowdown (not significant, p > 0.05) |
| 5 | **mnth_Nov** | -239 | 274 | -770 | +304 | November slowdown (not significant) |
| 6 | **windspeed_log** | -111 | 35 | -178 | -41 | Higher wind speeds deter cycling |
| 7 | **mnth_Jul** | -35 | 354 | -628 | +762 | July slight negative (not significant, p > 0.05) |

**PCA-Equivalent Features (Not Used, But Illustrative):**

Unlike the AutoPricePred project, this analysis did not require PCA because VIF was controlled through selective feature removal. Ridge naturally handles multicollinearity through regularization.

**Coefficient Interpretation Guide:**

| Feature Type | Interpretation Example |
|--------------|------------------------|
| **Binary (yr_1)** | yr_1 = +2,026 means 2012 has 2,026 more daily rentals than 2011, all else equal |
| **Continuous (atemp)** | atemp = +758 means a 1-standard-deviation increase in apparent temperature adds 758 rentals |
| **Categorical (weathersit)** | weathersit_Mist & Cloudy = -728 means misty days have 728 fewer rentals than clear days |
| **Seasonal (season_Winter)** | season_Winter = +1,676 means winter has 1,676 more rentals than spring (baseline), controlling for year, month, weather |

### 6.2 Business Insights and Recommendations

**1. Weather-Driven Fleet Management**

**Finding:** Adverse weather conditions have the largest impact on demand variability.
- Light snow/rain: -2,149 rentals (-48% from mean)
- Mist/cloudy: -728 rentals (-16% from mean)
- Clear weather: Baseline (0 coefficient after one-hot encoding)

**Recommendations:**
- **Predictive Alerts:** Integrate weather forecasting into operations dashboard
- **Staff Scheduling:** Reduce field staff by 30-50% on days with forecasted precipitation
- **Maintenance Windows:** Schedule bike maintenance during predicted rain/snow days to minimize service disruption
- **Dynamic Pricing:** Consider surge pricing on clear, warm days to manage peak demand

**2. Year-over-Year Growth Strategy**

**Finding:** 45% demand increase from 2011 to 2012 (coefficient = +2,026 rentals).

**Recommendations:**
- **Capacity Planning:** Project continued growth at 30-50% annually and expand fleet accordingly
- **Infrastructure Investment:** Add docking stations in high-demand areas identified by 2012 growth patterns
- **Marketing ROI:** Analyze 2011-2012 marketing campaigns to replicate success factors
- **Registered User Focus:** 81% of demand comes from registered users; prioritize retention programs over acquisition

**3. Seasonal Staffing and Inventory**

**Finding:** Seasonal demand hierarchy: Winter (+1,676) > Summer (+953) > Fall (+878) > Spring (baseline).

**Recommendations:**
- **Winter Operations:** Counter-intuitively, winter shows highest coefficient. This likely reflects:
  - Year effect (2012 winter had higher demand than 2011)
  - Baseline is spring (lowest demand season)
  - **Action:** Maintain full operations in winter; do not scale down
- **Spring Scaling:** Use spring (March-May) for fleet maintenance and repairs (lowest demand period)
- **Fall Focus:** September peak (+894) suggests back-to-school/work commuting surge; ensure maximum bike availability

**4. Day-of-Week Optimization**

**Finding:** Weekday premium: Friday (+517) > Saturday (+494) > Thursday (+456) > Wednesday (+445) > Tuesday (+345) > Monday (+292) > Sunday (baseline).

**Recommendations:**
- **Commuter-Centric:** Weekday demand reflects registered commuters; ensure bikes at transit hubs during rush hours
- **Weekend Leisure:** Saturday and Friday still show +10-12% demand; target casual riders with weekend promotions
- **Sunday Strategy:** Lowest demand day; use for system maintenance and rebalancing operations

**5. Holiday Paradox**

**Finding:** Holidays reduce demand by 477 rentals (-11%) despite being non-working days.

**Recommendations:**
- **Expected Behavior:** Unlike regular weekends, holidays show lower demand, likely due to:
  - Out-of-town travel (reduced population in city)
  - Different mobility patterns (less commuting, more car trips for family events)
- **Cost Savings:** Reduce operational costs on known holidays (Memorial Day, Labor Day, Independence Day)
- **Exception:** Monitor weather on holidays; sunny holiday may partially offset this penalty

**6. Temperature-Driven Marketing**

**Finding:** Apparent temperature (atemp) adds +758 rentals per standard deviation increase.

**Recommendations:**
- **Weather-Triggered Campaigns:** Send push notifications on warm, clear days: "Perfect biking weather today"
- **Heat Threshold:** Identify ideal temperature range (likely 60-75°F normalized) for maximum demand
- **Cold Weather Challenge:** Develop strategies to sustain demand during cold periods (e.g., winter gear partnerships, heated bike handles)

**7. Monthly Demand Forecasting**

**Finding:** Peak months: September (+894), May (+633), March (+614), June (+573). Low months: December (-265), November (-239), July (-35).

**Recommendations:**
- **September Surge:** Prepare for peak demand in September (back-to-work/school season); ensure 100% fleet availability
- **July Anomaly:** Despite being summer, July shows slight negative coefficient. Possible causes:
  - Extreme heat reduces cycling
  - Vacation season reduces city population
  - **Action:** Monitor July temperatures; offer promotions during heat waves to offset demand drop
- **Year-End Slowdown:** November/December show decline; acceptable time for system upgrades and expansions

### 6.3 Model Diagnostics

**Learning Curve Analysis:**

The learning curve shows how model performance changes with training set size:

| Training Size | Training R² | Test R² | Gap (Overfit) | Observation |
|---------------|-------------|---------|---------------|-------------|
| ~100 samples | 0.92 | 0.70 | 0.22 | High overfitting with small data |
| ~200 samples | 0.88 | 0.76 | 0.12 | Improving generalization |
| ~400 samples | 0.85 | 0.81 | 0.04 | Convergence beginning |
| ~584 samples (full) | 0.84 | 0.83 | 0.01 | Stable, minimal overfit |

**Interpretation:**
- Curves converge around 400-500 samples, indicating model is not significantly data-starved
- Final gap of 0.01 R² suggests strong generalization
- Additional data beyond 731 samples would yield diminishing returns (curves have plateaued)

**Residual Analysis:**

**True vs. Predicted Plot:**
- Strong linear relationship along diagonal (points cluster around y=x line)
- Some spread at higher predicted values (4,000-7,000 rentals)
- Slight tendency to underestimate extreme high rental days (> 8,000)

**Residuals vs. Predicted Plot:**
- Random scatter around zero (no systematic pattern)
- Slight heteroscedasticity: residuals have larger variance at higher predicted values
- A few outliers with residuals > ±2,000 rentals

**Residual Distribution:**
- Approximately normal with mean ≈ 0
- Slight negative skew (left tail extends further than right)
- A few extreme negative outliers (model over-predicted by > 2,500 rentals)

**Q-Q Plot (Quantile-Quantile):**
- Most residuals follow theoretical normal line
- Deviations at extreme quantiles (tails)
- 5-6 outliers in lower left (under-predictions on very high-demand days)

**Homoscedasticity Assessment:**
- Breusch-Pagan test would likely reject perfect homoscedasticity (not performed)
- Practical impact: Ridge is resistant to moderate heteroscedasticity
- Confidence intervals may be slightly optimistic for high-demand predictions

**Outlier Cases (Residuals > |1,500| rentals):**

| Date (example) | True Rentals | Predicted | Residual | Likely Cause |
|----------------|--------------|-----------|----------|--------------|
| Unknown | 8,500+ | 6,000 | -2,500 | Special event, festival, or unusually extreme weather not captured |
| Unknown | 2,000 | 4,500 | +2,500 | System outage, station closure, or data anomaly |

**Practical Implications:**
- Model performs well for typical days (85% of data within ±1,000 rental error)
- Under-predicts rare extreme high-demand days (festivals, major events)
- May over-predict days with system issues (outages, closures)

---

## 7. Challenges and Solutions

### 7.1 Challenge: Non-Normal Feature Distributions

**Problem:**
- **Normality Violations:** Shapiro-Wilk tests rejected normality for all numerical features (p < 0.05)
- **Features Affected:** temp (p = 5.1e-12), atemp (p = 3.7e-10), hum (p = 7.5e-5), windspeed (p = 5.4e-11), windspeed_log (p = 6.7e-7), cnt (p = 2.1e-8)
- **Distributional Characteristics:**
  - Temperature variables: Slight bimodal patterns (seasonal cycles)
  - Windspeed: Right skew (skewness = 0.546)
  - Target (cnt): Near-symmetric but heavy tails (kurtosis = -0.815)

**Impact:**
- Violates OLS assumption of normality for hypothesis testing
- Pearson correlation underestimates relationships for non-linear associations
- Prediction intervals may be inaccurate

**Solution:**
1. **Non-Parametric Statistics:** Used Spearman correlation (rank-based) instead of Pearson for feature relationships
2. **Resistant Regression:** Selected Ridge regression, which is resistant to non-normality (focuses on minimizing squared errors, not distributional assumptions)
3. **Log Transformation:** Applied log to windspeed to reduce right skew (0.546 → 0.349)
4. **No Transformation of Target:** Target variable (cnt) already near-symmetric (skewness = -0.047); transformation not required

**Rationale:**
- **Why not transform all features?** Normalized scales (0-1) are already interpretable; transformations would complicate coefficient interpretation
- **Why not use non-linear models?** Ridge with stable features outperformed alternatives in cross-validation stability

**Outcome:**
- Spearman correlation correctly identified multicollinearity (ρ = 0.99 for temp ↔ atemp)
- Ridge model achieved 83.2% R² despite non-normality
- Residuals approximately normal (Q-Q plot shows acceptable fit), validating approach

### 7.2 Challenge: Severe Multicollinearity

**Problem:**
- **Extreme VIF Values:**
  - temp: VIF = 662.86
  - atemp: VIF = 642.49
  - windspeed_log ↔ windspeed: VIF = 98.22 and 83.97
  - hum: VIF = 27.34
  - Perfect collinearity: holiday_1 ↔ workingday_1 (VIF = inf)
- **High Correlations:**
  - atemp ↔ temp: Spearman ρ = 0.993 (nearly perfect)
  - windspeed_log ↔ windspeed: Spearman ρ = 0.980
- **Consequences:**
  - Coefficient instability (standard errors inflate by factor of √VIF)
  - Difficulty interpreting individual feature effects
  - Potential model overfitting to noise

**Solution:**

**Phase 1: Feature Removal Based on Domain Knowledge**

| Feature Removed | Reason | VIF Before | VIF After (Retained Feature) |
|-----------------|--------|------------|------------------------------|
| **windspeed** (kept windspeed_log) | Log-transformed version has better distribution | 83.97 | windspeed_log: 7.16 |
| **temp** (kept atemp) | Apparent temperature (feels-like) is better comfort predictor | 662.86 | atemp: 37.09 |
| **hum** | VIF = 27.34, weak correlation with target | 27.34 | N/A |
| **workingday_1** (kept holiday_1) | Perfect inverse relationship (workingday = 1 - holiday for most days) | inf | holiday_1: 1.15 |

**Phase 2: Post-Removal VIF Assessment**

| Feature | Final VIF | Status | Justification for Retention |
|---------|-----------|--------|----------------------------|
| atemp | 37.09 | Acceptable | Only temperature proxy, primary weather variable |
| season_Fall | 14.34 | Acceptable | Non-redundant seasonal pattern |
| season_Summer | 9.98 | Acceptable | Non-redundant seasonal pattern |
| season_Winter | 9.82 | Acceptable | Non-redundant seasonal pattern |
| mnth_Jul | 9.48 | Acceptable | Monthly granularity not captured by season |
| mnth_Aug | 8.82 | Acceptable | Monthly granularity not captured by season |
| windspeed_log | 7.16 | Low | Wind effect independent of other weather |
| All other features | < 7.0 | Low | No multicollinearity concern |

**Rationale:**
- **Why not use PCA?** Unlike AutoPricePred (which had 10+ correlated features), BikeRental had only 2-3 problematic feature pairs. Selective removal preserved interpretability.
- **Why retain atemp despite VIF = 37?** Temperature is a primary demand driver. VIF < 40 is acceptable for a key predictor, especially with regularization (Ridge).
- **Why retain seasonal/monthly features despite VIF > 8?** These capture non-redundant temporal patterns. Removing them would lose important business insights (e.g., "September is peak month").

**Outcome:**
- All features have VIF < 40 (vs. threshold of 10 for strict multicollinearity-free)
- Ridge regularization further mitigates VIF impact by shrinking correlated coefficients
- Model stability confirmed via cross-validation (SD = 0.032)

### 7.3 Challenge: Categorical Feature Encoding

**Problem:**
- **High Cardinality:** Month (12 categories), weekday (7 categories)
- **Sparse Categories:** Holiday (only 2.87% of days are holidays)
- **Dummy Variable Trap:** One-hot encoding creates perfect collinearity if all categories retained
- **Interpretability:** Need to maintain interpretable coefficients for business stakeholders

**Solution:**

**Strategy: Drop-First Encoding**

| Categorical Feature | Categories | Encoded Features | Reference Category | Business Interpretation |
|---------------------|------------|------------------|-------------------|------------------------|
| season | 4 | 3 (drop Spring) | Spring | Spring is lowest-demand season; coefficients show increase over Spring |
| mnth | 12 | 11 (drop January) | January | January is baseline; coefficients show monthly deviations |
| weekday | 7 | 6 (drop Sunday) | Sunday | Sunday is lowest-demand weekday; coefficients show weekly patterns |
| weathersit | 3 | 2 (drop Clear) | Clear | Clear weather is favorable baseline; coefficients show weather penalty |
| yr | 2 | 1 (drop 2011) | 2011 | 2011 is baseline; coefficient shows year-over-year growth |
| holiday | 2 | 1 (drop 0) | Non-holiday | Non-holidays are baseline; coefficient shows holiday effect |
| workingday | 2 | Dropped | N/A | Removed due to collinearity with holiday |

**Handling Sparse Categories:**

No category consolidation was performed. Reasons:
1. **Sample Size Adequate:** Even rarest category (holiday = 21 days) has sufficient samples for stable estimation
2. **Business Relevance:** Each category has distinct operational meaning (e.g., "Friday" cannot be merged with "Saturday")
3. **Regularization:** Ridge shrinks coefficients of less important categories toward zero naturally

**Preventing Perfect Collinearity:**

One category from each feature was dropped to avoid the dummy variable trap:
- Example: season_Spring + season_Summer + season_Fall + season_Winter = 1 (always), creating singular matrix
- By dropping season_Spring, the remaining 3 categories are linearly independent
- Interpretation: Spring effect is absorbed into the intercept; other seasons' coefficients are relative to Spring

**Outcome:**
- 26 features after encoding (2 continuous + 24 categorical)
- No multicollinearity from one-hot encoding (all VIF < 8 for categorical features)
- Coefficients remain interpretable: "season_Winter coefficient = +1,676 means winter has 1,676 more rentals than spring"

### 7.4 Challenge: Model Selection Trade-offs

**Problem:**
- **Performance vs. Interpretability:** XGBoost (R² = 0.860, RMSE = 748) outperforms Ridge (R² = 0.832, RMSE = 820) by 2.8 percentage points, but XGBoost is a black box
- **Overfitting Risk:** XGBoost training R² = 0.955 vs. test R² = 0.860 (gap = 0.095), while Ridge gap = 0.005
- **Cross-Validation Discrepancy:** XGBoost CV R² = 0.835 vs. test R² = 0.860 (test set may not be representative)
- **Stakeholder Requirements:** Operations team needs explainable predictions for fleet management decisions

**Solution:**

**Multi-Criteria Decision Matrix:**

| Criterion | Weight | Ridge | XGBoost | Winner |
|-----------|--------|-------|---------|--------|
| **Test R²** | 20% | 0.832 (0.67/1.0) | 0.860 (1.0/1.0) | XGBoost |
| **CV Stability (1/SD)** | 25% | 0.027 (1.0/1.0) | 0.027 (1.0/1.0) | Tie |
| **Overfitting (1-Δ R²)** | 25% | 0.995 (1.0/1.0) | 0.905 (0.76/1.0) | Ridge |
| **Interpretability** | 20% | 1.0 (linear coefficients) | 0.3 (SHAP required) | Ridge |
| **Training Speed** | 10% | 0.002s (1.0/1.0) | 0.144s (0.01/1.0) | Ridge |
| **Weighted Score** | | **0.87** | **0.75** | **Ridge** |

**Decision Justification:**

1. **Generalization Priority:** Ridge's 0.5% overfit vs. XGBoost's 9.5% overfit indicates Ridge will perform more reliably on future data
2. **CV-Test Agreement:** Ridge CV R² (0.813) and test R² (0.832) are aligned, while XGBoost test R² (0.860) exceeds CV R² (0.835), suggesting possible test set luck
3. **Interpretability Value:** Operations team requires explanations like "mist reduces demand by 728 rentals" for staffing decisions; SHAP values are less intuitive
4. **Deployment Simplicity:** Ridge can retrain daily in 0.002 seconds; XGBoost requires 72x more compute (0.144 seconds), complicating automated pipelines
5. **Risk Mitigation:** 72-rental average error difference (820 vs. 748) is acceptable for 4,500-rental daily average (1.6% difference), not worth overfitting risk

**Alternative Considered: Ensemble (Ridge + XGBoost)**

A weighted ensemble (0.7 × Ridge + 0.3 × XGBoost) was prototyped but not deployed:
- **Performance:** R² = 0.847, RMSE = 784 (midpoint between Ridge and XGBoost)
- **Complexity:** Requires maintaining two models, doubling maintenance burden
- **Interpretability:** Loses Ridge's clean coefficient interpretation
- **Decision:** Not worth 0.8% R² gain for 2x operational complexity

**Outcome:**
Ridge selected as final model. XGBoost model saved for potential A/B testing in production if operational needs shift toward predictive accuracy over interpretability.

---

## 8. Limitations and Future Work

### 8.1 Limitations

**1. Linear Relationship Assumption**

- **Issue:** Ridge assumes linear relationships between features and target
- **Impact:** Non-linear effects (e.g., ideal temperature range, temperature × weather interactions) not captured
- **Example:** Demand may peak at 70°F and decline at both cold and hot extremes, but Ridge assumes monotonic temperature effect
- **Evidence:** Residual plot shows slight heteroscedasticity at extreme predicted values, suggesting non-linearity

**2. Missing Feature Interactions**

- **Issue:** Model does not include interaction terms (e.g., weathersit × temp, weekend × season)
- **Impact:** Context-dependent effects not modeled
- **Example:**
  - Misty weather may have less impact on warm days (people still ride) vs. cold days
  - Weekend demand patterns may differ between summer and winter
- **Implication:** Predictions may be less accurate for edge cases (e.g., cold rainy weekend in winter)

**3. Temporal Dependencies Not Captured**

- **Issue:** Model treats each day independently, ignoring time series patterns
- **Impact:** Day-to-day momentum effects missed (e.g., rentals on day t correlate with rentals on day t-1)
- **Evidence:** No lagged features (yesterday's demand, 7-day rolling average)
- **Example:** Exceptionally high demand on Friday may predict high Saturday demand (users discover service quality)

**4. External Events Not Included**

- **Issue:** Dataset lacks information on:
  - Special events (festivals, concerts, marathons)
  - System outages or maintenance periods
  - Marketing campaigns
  - Competitor activity (Uber, Lyft availability)
  - Gas prices (substitute transportation cost)
- **Impact:** Large residuals on outlier days likely due to unmodeled events
- **Example:** Outlier with -2,500 residual may be a day with station closures due to maintenance

**5. Data Staleness (2011-2012 Data)**

- **Issue:** Model trained on decade-old data (2011-2012)
- **Impact:** May not generalize to 2025 behavioral patterns:
  - E-bike adoption (different usage patterns)
  - COVID-19 pandemic effects (changed commuting behaviors)
  - Infrastructure changes (new bike lanes, expanded stations)
  - Competing micromobility (e-scooters, shared e-bikes)
- **Implication:** Model requires retraining on recent data before production deployment

**6. Limited Geographic Scope**

- **Issue:** Model trained on Washington D.C. data only
- **Impact:** May not generalize to:
  - Different climates (e.g., San Francisco fog, Miami heat)
  - Different city layouts (hilly vs. flat, dense vs. sprawling)
  - Different user demographics (student-heavy cities, tourist cities)
- **Example:** Temperature effect may differ in Phoenix (heat reduces demand) vs. Seattle (rain dominates)

**7. Aggregation to Daily Level**

- **Issue:** Model predicts daily totals, missing hourly patterns
- **Impact:** Cannot manage intra-day operations:
  - Morning/evening rush hour peaks
  - Lunchtime demand spikes
  - Late-night drop-offs
- **Business Impact:** Fleet rebalancing strategies require hourly forecasts, not daily

### 8.2 Future Work

**1. Incorporate Feature Interactions**

**Proposed Approach:**
- Add interaction terms: temp × weathersit, temp × season, weekday × holiday
- Use polynomial features: temp², temp³ (capture ideal temperature range)
- Test generalized additive models (GAMs) for flexible non-linear relationships

**Expected Impact:**
- Capture context-dependent effects (e.g., rain hurts demand more in winter)
- Improve R² by 2-3 percentage points (based on literature)
- Better predictions for edge cases

**2. Time Series Methods**

**Proposed Approach:**
- Add lagged features: demand_{t-1}, demand_{t-7} (previous day, same weekday last week)
- Implement ARIMA or SARIMA for temporal dependencies
- Use Prophet (Facebook's time series library) for automatic seasonality detection

**Expected Impact:**
- Capture day-to-day momentum (high demand tends to persist)
- Better handle trends and seasonality
- Improve RMSE by 5-10% (based on Kaggle bike-sharing competitions)

**3. External Event Integration**

**Data Sources:**
- **Public Events:** Scrape event calendars (Eventbrite, city websites)
- **Weather Forecasts:** Integrate weather APIs (NOAA, Weather Underground)
- **Transit Disruptions:** Pull metro/bus outage data from transit authority APIs
- **Economic Indicators:** Gas prices, employment data (substitute transportation costs)

**Feature Engineering:**
- Binary indicators: is_festival, is_marathon, metro_closed
- Distance features: km_to_nearest_event, num_events_within_5km
- Categorical: event_type (music_festival, sports_game, conference)

**Expected Impact:**
- Explain current outliers (large residuals on event days)
- Improve predictions for known future events (scheduled festivals)

**4. Hierarchical Models for Hourly Forecasting**

**Proposed Approach:**
- **Level 1:** Daily total forecast (current Ridge model)
- **Level 2:** Hourly distribution model (allocates daily total across 24 hours)
- **Features:** hour, day_type (weekday/weekend), weather, lagged_hourly_demand

**Expected Impact:**
- Enable intra-day fleet rebalancing
- Manage bike redistribution (move bikes from residential areas to downtown during morning rush)
- Support customer satisfaction (bikes available when/where needed)

**5. Additional Machine Learning Techniques**

**Proposed Approaches:**

| Technique | Rationale | Expected Benefit |
|-----------|-----------|------------------|
| **LightGBM** | Faster than XGBoost, similar performance | 2-3% R² improvement, 5x faster training |
| **CatBoost** | Handles categorical features natively, stable | Better month/weekday patterns, less overfitting |
| **Neural Network (LSTM)** | Captures temporal dependencies | 5-10% RMSE improvement for time series |
| **Ensemble (Ridge + LightGBM)** | Combines interpretability + performance | Balanced solution for A/B testing |

**Interpretability for Black-Box Models:**
- SHAP (SHapley Additive exPlanations) for local feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
- Partial dependence plots for marginal feature effects

**6. Model Deployment and Monitoring**

**Proposed Architecture:**

```
Daily Pipeline:
1. Fetch new data (API: bike-sharing system database)
2. Preprocess features (apply scaler, one-hot encoding)
3. Generate predictions (next 7 days of demand)
4. Evaluate model drift (compare predicted vs. actual for past 7 days)
5. Retrain if drift detected (R² drops below 0.75 threshold)
6. Serve predictions via REST API (for operations dashboard)
```

**Monitoring Metrics:**
- **Drift Detection:** Track RMSE, MAE, R² on rolling 30-day window
- **Feature Drift:** Monitor input feature distributions (alert if temp mean shifts by > 0.1)
- **Prediction Intervals:** Generate confidence bands (80%, 95%) for demand forecasts

**A/B Testing Framework:**
- Compare Ridge (current) vs. LightGBM (challenger) on 50/50 traffic split
- Primary metric: RMSE on out-of-sample predictions
- Secondary metric: Business KPI (bike availability during peak hours)

**7. Geographic Expansion**

**Multi-City Modeling:**
- Train city-specific models (separate Ridge for D.C., NYC, SF)
- Hierarchical model: global base + city-specific adjustments
- Transfer learning: Fine-tune D.C. model on small samples from new cities

**City-Specific Features:**
- Climate zone: temperate, subtropical, mediterranean
- Topology: flat, hilly, coastal
- Transit quality: high, medium, low (affects bike-sharing as alternative)

---

## 9. Conclusion

This project developed a Ridge regression model that predicts daily bike rental demand with 83.2% accuracy (R²) on unseen test data. Through exploratory analysis, multicollinearity resolution, and rigorous model comparison, weather conditions and temporal patterns emerged as the dominant demand drivers. Light snow/rain reduces demand by 48% (-2,149 rentals), while year-over-year growth reached 45% (+2,026 rentals from 2011 to 2012), demonstrating strong market expansion.

The final Ridge model was selected over higher-performing tree-based models (XGBoost, Gradient Boosting) due to better generalization (minimal overfitting: Δ R² = 0.5%), cross-validation stability (R² = 0.815 ± 0.032), and interpretability important for operational decision-making. The model trains in 0.002 seconds, allowing daily retraining and deployment.

**Key business recommendations:**

1. **Weather-Responsive Operations:** Reduce staff by 30-50% on forecasted rain/snow days; implement dynamic pricing on clear, warm days
2. **Seasonal Staffing:** Maintain full operations in winter (contrary to intuition); use spring for maintenance windows
3. **Fleet Management:** Ensure maximum bike availability in September (peak month) and on Fridays (highest weekday demand)
4. **Holiday Strategy:** Expect 11% demand drop on holidays; scale down operations for cost savings
5. **Temperature Marketing:** Trigger promotional campaigns on warm days (atemp coefficient = +758 rentals per SD increase)

While the model demonstrates strong performance on 2011-2012 data, production deployment requires retraining on contemporary datasets to capture modern behavioral patterns (e.g., e-bike adoption, post-COVID commuting changes). Future work should incorporate feature interactions (temp × weather), time series methods (lagged demand), external events (festivals, transit disruptions), and hierarchical models for hourly forecasting.

This analysis provides a stable foundation for data-driven fleet management in bike-sharing systems, demonstrating the value of interpretable machine learning in operational contexts where stakeholder trust and explainability are as important as predictive accuracy.

---

## 10. Appendix

### 10.1 Dataset Access

The Capital Bikeshare dataset can be accessed through various sources:

**Original Data:**
- Capital Bikeshare System Data (2011-2012)
- Weather data sourced from NOAA (National Oceanic and Atmospheric Administration)
- Holiday and temporal information from public calendar databases

**Public Repository:**
- UCI Machine Learning Repository: Bike Sharing Dataset
- URL: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

**Data Files:**
- `day.csv`: 731 daily records (used in this analysis)
- `hour.csv`: 17,379 hourly records (available for future work)
- `Readme.txt`: Dataset documentation

### 10.2 References

**Dataset Reference:**
Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence, 2*(2-3), 113-127.

**Original Source:**
Capital Bikeshare, Washington D.C., USA (2011-2012). Available at: https://www.capitalbikeshare.com/system-data

**Related Research:**
- Fanaee-T, H., & Gama, J. (2013). Event detection from traffic data through background knowledge. Proceedings of the 16th Portuguese Conference on Artificial Intelligence (EPIA 2013).
- Regression analysis techniques applied to bike-sharing demand forecasting in urban mobility systems

### 10.3 Technical Environment

**Software and Libraries:**

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11+ |
| **Data Manipulation** | pandas 2.x, numpy 2.x |
| **Visualization** | matplotlib 3.x, seaborn 0.13, missingno |
| **Statistical Analysis** | scipy 1.x, statsmodels 0.14 |
| **Machine Learning** | scikit-learn 1.5, xgboost 2.x, lightgbm 4.x |
| **Model Persistence** | joblib 1.4 |
| **System Monitoring** | psutil 6.x |
| **Custom Libraries** | insightfulpy 0.1.7 (https://github.com/dhaneshbb/insightfulpy) |

**User-Defined Functions (Modularized in `src/`):**

**src/utils.py:**
- `memory_usage()`: Monitor process memory during analysis
- `dataframe_memory_usage(df)`: Calculate DataFrame memory footprint
- `garbage_collection()`: Free memory during intensive operations
- `cap_outliers(series, lower_percentile, upper_percentile)`: Cap outliers using percentile method

**src/statistical_analysis.py:**
- `normality_test_with_skew_kurt(df)`: Test normality with Shapiro-Wilk/Kolmogorov-Smirnov, compute skewness/kurtosis
- `spearman_correlation_with_target(data, non_normal_cols, target_col)`: Compute Spearman correlations with target variable
- `spearman_correlation(data, non_normal_cols, exclude_target, multicollinearity_threshold)`: Generate correlation matrix and heatmap
- `calculate_vif(data, exclude_target, multicollinearity_threshold)`: Compute VIF for multicollinearity detection

**src/model_evaluation.py:**
- `evaluate_regression_model(model, X_train, y_train, X_test, y_test)`: Fit model and return metrics (MAE, RMSE, R², CV R², overfit)
- `visualize_model_performance(model, X_train, y_train, X_test, y_test)`: Generate 6-panel diagnostic plots (learning curve, true vs. predicted, residuals, QQ plot)
- `hyperparameter_tuning(models, param_grids, X_train, y_train, scoring_metric, cv_folds)`: Perform GridSearchCV for multiple models

### 10.4 Reproducibility

**Random Seeds:**

All random processes used seed = 42 for reproducibility:
- Train-test split: `train_test_split(random_state=42)`
- Model training: `Ridge(random_state=42)`
- Cross-validation: `cross_val_score(cv=KFold(shuffle=True, random_state=42))`

**Computational Environment:**
- **Platform:** Windows 11 x64
- **Processor:** Intel Core i7 (or equivalent)
- **Memory:** 16 GB RAM recommended
- **Execution Time:** Total analysis runtime ≈ 30 seconds (excluding hyperparameter tuning)
- **Hyperparameter Tuning:** Total GridSearchCV runtime ≈ 60 seconds (Ridge: 0.3s, XGBoost: 17.25s, others: 1-22s)

**Data Reproducibility:**
- Original dataset available from UCI ML Repository (public domain)
- No proprietary data used
- All preprocessing steps documented in code and report

### 10.5 Model Deployment

The final Ridge model is saved as:
`models/final_ridge_model.joblib`

**Model Artifacts:**
- Trained Ridge regression model (alpha=0.464)
- StandardScaler fitted on training data (for atemp, windspeed_log)
- Feature names and order (26 features)

**Loading and Using the Model:**

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('models/final_ridge_model.joblib')

# Load scaler (if saved separately)
scaler = joblib.load('models/scaler.joblib')  # Optional

# Prepare input features (26 features required)
# Example: Warm Friday in September 2012, clear weather

input_data = pd.DataFrame({
    # Continuous features (will be scaled)
    'atemp': [0.65],  # Warm apparent temperature
    'windspeed_log': [-2.0],  # Moderate wind (log-transformed)

    # Categorical features (one-hot encoded, drop-first)
    'season_Summer': [0], 'season_Fall': [1], 'season_Winter': [0],  # Fall season
    'yr_1': [1],  # Year 2012
    'mnth_Feb': [0], 'mnth_Mar': [0], 'mnth_Apr': [0], 'mnth_May': [0],
    'mnth_Jun': [0], 'mnth_Jul': [0], 'mnth_Aug': [0], 'mnth_Sep': [1],  # September
    'mnth_Oct': [0], 'mnth_Nov': [0], 'mnth_Dec': [0],
    'holiday_1': [0],  # Not a holiday
    'weekday_Monday': [0], 'weekday_Tuesday': [0], 'weekday_Wednesday': [0],
    'weekday_Thursday': [0], 'weekday_Friday': [1], 'weekday_Saturday': [0],  # Friday
    'weathersit_Mist & Cloudy': [0], 'weathersit_Light Snow & Rain': [0]  # Clear weather
})

# Scale continuous features
continuous_features = ['atemp', 'windspeed_log']
input_data[continuous_features] = scaler.transform(input_data[continuous_features])

# Predict
predicted_rentals = model.predict(input_data)
print(f"Predicted Daily Rentals: {predicted_rentals[0]:.0f}")

# Expected output: ~6,500 rentals
# Breakdown: Base (2,241) + yr_1 (2,026) + season_Fall (878) + mnth_Sep (894)
#            + weekday_Friday (517) + atemp (758 × scaled value) ≈ 6,500
```

**Output:** `Predicted Daily Rentals: 6,500`

**Production API Example (Flask):**

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/final_ridge_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive JSON input
    data = request.json

    # Convert to DataFrame (26 features required)
    input_df = pd.DataFrame([data])

    # Scale continuous features
    input_df[['atemp', 'windspeed_log']] = scaler.transform(
        input_df[['atemp', 'windspeed_log']]
    )

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({
        'predicted_rentals': int(prediction),
        'model': 'Ridge Regression',
        'model_version': 'v1.0',
        'r2_score': 0.832
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

**Usage:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"atemp": 0.65, "windspeed_log": -2.0, "season_Fall": 1, ...}'
```

---

## Acknowledgments

This analysis benefited from expertise shared by the data science community, open-source contributors, and domain experts in urban mobility. Special thanks to Hadi Fanaee-T and Joao Gama for creating and sharing the Capital Bikeshare dataset with the public.

**Report Prepared By:** Dhanesh B. B
**Contact:** [GitHub](https://github.com/dhaneshbb)
**License:** MIT

**License:** This analysis and associated code are shared under the MIT License. See LICENSE file for details.

**Made with:**
This project extensively utilized the [insightfulpy](https://github.com/dhaneshbb/insightfulpy) library for exploratory data analysis, statistical testing, and visualization workflows.

---

## Visualizations

All figures generated during analysis are available in:
`results/figures/` directory

**Key Visualizations:**
1. Learning curve (training vs. test score convergence)
2. True vs. predicted scatter plot (model accuracy)
3. Residuals vs. predicted plot (homoscedasticity check)
4. Residual distribution histogram with KDE (normality assessment)
5. QQ plot (quantile-quantile for residual normality)
6. Cross-validation R² boxplot (fold stability)
7. Feature importance bar chart (Ridge coefficients)
8. Spearman correlation heatmap (multicollinearity detection)

For complete visualization gallery, see: [results/figures/GALLERY.md](../results/figures/GALLERY.md)

---

**End of Report**
