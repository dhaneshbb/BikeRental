# Table of Contents

- [Data Analysis](#data-analysis)
    - [Import data](#import-data)
    - [Imports & functions](#imports-functions)
  - [Data understanding](#data-understanding)
    - [day](#day)
    - [hr](#hr)
    - [final_data](#final-data)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Univariate Analysis](#univariate-analysis)
      - [num_analysis](#num-analysis)
      - [cat_analysis](#cat-analysis)
    - [Bivariate & Multivariate Analysis](#bivariate-multivariate-analysis)
- [Predictive Model](#predictive-model)
  - [Preprocessing](#preprocessing)
      - [Encoding](#encoding)
      - [Relation](#relation)
      - [Splitting](#splitting)
  - [Model Development](#model-development)
    - [Model Training & Evaluation](#model-training-evaluation)
    - [Model comparision & Interpretation](#model-comparision-interpretation)
    - [Best Model](#best-model)
    - [Saving the Model](#saving-the-model)
    - [Loading the Model Further use](#loading-the-model-further-use)
- [Table of Contents](#table-of-contents)
- [Acknowledgment](#acknowledgment)
- [Report](#report)
- [References](#references)
- [Appendix](#appendix)
  - [About data](#about-data)
  - [Source Code and Dependencies](#source-code-and-dependencies)
  - [Code Repository](#code-repository)


---

# Acknowledgment  

I would like to express my sincere gratitude to mentors, colleagues, peers, and the data science community for their unwavering support, constructive feedback, and encouragement throughout this project. Their insights, shared expertise, and collaborative spirit have been invaluable in overcoming challenges and refining my approach. I deeply appreciate the time and effort they have dedicated to helping me achieve success in this endeavor

---

# Report

**Final Data Analysis Report**

 Executive Summary  
This report analyzes bike rental data from two datasets (day.csv and hour.csv) to understand patterns in bike-sharing demand. Key findings include strong seasonal trends, correlations between weather variables and rentals, and actionable insights for predictive modeling.  



 1. Data Overview  
 1.1 Dataset Description  
- Daily Data: 731 records, 16 attributes (e.g., dteday, temp, cnt).  
- Hourly Data: 17,379 records, 17 attributes (includes hourly rental counts).  
- Key Variables:  
  - Temporal: dteday, season, mnth, weekday.  
  - Weather: temp, hum, windspeed.  
  - Usage: casual, registered, cnt (total rentals).  

 1.2 Data Quality  
- No missing values, duplicates, or infinite values in either dataset.  
- Outliers: Detected in hum (2) and windspeed (13), addressed by capping at percentiles.  
- High Cardinality: dteday uniquely identifies dates.  

 1.3 Merged Dataset  
- Combined daily summaries from hourly data (e.g., mean temperature, mode weather) with original daily data.  
- Final Features: 16 columns, including aggregated metrics and cleaned outliers.  



 2. Data Cleaning & Transformation  
 2.1 Outlier Handling  
- Capping: Outliers in hum and windspeed capped at 1st/99th percentiles.  
- Log Transformation: Applied to windspeed to normalize distribution.  

 2.2 Feature Engineering  
- Categorical Encoding:  
  - season → Spring, Summer, Fall, Winter.  
  - weathersit → Clear, Mist, Rain/Snow.  
  - One-hot encoding applied to mnth, weekday, and other categorical variables.  
- Redundant Features Removed:  
  - casual, registered (sum to cnt).  
  - temp, windspeed (highly correlated with atemp, windspeed_log).  



 3. Exploratory Data Analysis (EDA)  
 3.1 Temporal Trends  
- Seasonality: Rentals peak in summer/fall (avg. 5,956/day) and drop in winter (avg. 3,152/day).  
- Yearly Growth: 50% increase in rentals from 2011 to 2012.  
- Daily Patterns: Higher demand on weekdays (working days).  

 3.2 Weather Impact  
- Temperature: Strong positive correlation with rentals (*r* = 0.63).  
- Humidity/Wind: Negative correlations (*r* = -0.23 and -0.12, respectively).  

 3.3 Categorical Insights  
- Weather: 63% of days are clear, correlating with 25% higher rentals.  
- Holidays: 2.87% of days; rentals 15% lower than non-holidays.  


 4. Multicollinearity & Feature Selection  
 4.1 Correlation Analysis  
- High Correlations:  
  - temp vs. atemp (*r* = 0.99).  
  - windspeed vs. windspeed_log (*r* = 0.98).  
- Action: Removed temp and windspeed to reduce redundancy.  

 4.2 VIF Analysis  
- High VIF Features: season, mnth, atemp (VIF > 8).  
- Final Features: 20 variables retained after addressing multicollinearity.  



 5. Preprocessing for Modeling  
 5.1 Dataset Preparation  
- Encoded Variables: 15 categorical features converted to dummy variables.  
- Target Variable: cnt (total rentals).  

 5.2 Key Features for Prediction  
- Top Predictors:  
  1. atemp (normalized temperature).  
  2. season_Fall (high-demand season).  
  3. weathersit_Clear.  



 6. Conclusion & Recommendations  
 6.1 Insights  
1. Demand Drivers: Temperature and clear weather significantly boost rentals.  
2. Seasonality: Invest in bike availability during summer/fall.  
3. Weather Alerts: Reduce stock during high humidity/wind days.  

 6.2 Next Steps  
- Predictive Modeling: regression to forecast demand.  

Appendix: Full code and visualizations available in the project repository. 

---

**Final Model Comparison and Report**


OLS Regression Model Overview  
- R²: 0.838 (training), 0.834 (test)  
- Key Features:  
  - Positive Impact: Apparent temperature (atemp), seasonal effects (Winter > Summer/Fall), and year-over-year growth (yr_1).  
  - Negative Impact: Adverse weather conditions (weathersit_Light Snow & Rain), wind speed (windspeed_log).  
- Test Performance:  
  - RMSE: 816.66  
  - Interpretability: High (linear coefficients).  
- Limitations: Slight non-normality in residuals but no autocorrelation (Durbin-Watson = 1.98).  



 1. Model Comparison Summary

| Metric              | Linear Models (Base) | Tree-Based (Base) | Tuned Linear Models | Tuned Tree-Based      |
|-------------------------|--------------------------|-----------------------|-------------------------|---------------------------|
| Best R²             | 0.834 (Linear Reg)       | 0.856 (Gradient Boosting) | 0.833 (Ridge)      | 0.860 (XGBoost)       |
| Best MAE            | 593 (Lasso)              | 511 (Gradient Boosting)   | 595 (Ridge)        | 500 (XGBoost)         |
| Best RMSE           | 817 (Lasso)              | 760 (Gradient Boosting)   | 820 (Ridge)        | 748 (XGBoost)         |
| Overfit (Δ R²)      | 0.004–0.006              | 0.075–0.089              | 0.005–0.007        | 0.094–0.122               |
| Training Time       | <0.03s                   | 0.16–0.54s               | <0.02s             | 0.16–0.54s                |
| Interpretability    | High                     | Low                     | High                | Low                       |



 2. Final Model Selection
A. Best Interpretable Model: Tuned Ridge Regression  
- R²: 0.832  
- RMSE: 819.62  
- Key Strengths:  
  - Minimal overfitting (Δ R² = 0.005)  
  - Instant training time (0.002s)  
  - Clear coefficient-based interpretation  

B. Best Performance Model: Tuned XGBRegressor  
- R²: 0.860  
- RMSE: 748.11  
- Key Strengths:  
  - 14% lower RMSE than Ridge  
  - Handles non-linear relationships  
  - Robust to outliers  



 3. Final Model Evaluation (Ridge Regression)

| Metric | Value       |
|------------|-----------------|
| MSE        | 671,778.03      |
| RMSE       | 819.62          |
| R²         | 0.832           |
| Cross-Val R² | 0.815 ± 0.032 |

Residual Analysis:  
- Residuals are normally distributed with mean ≈ 0.  
- No systematic bias in predictions.  



 4. Feature Importance (Ridge Coefficients)

| Top Features                  | Impact                     |
|-----------------------------------|--------------------------------|
| weathersit_Light Snow & Rain    | Reduces rentals by 2,149   |
| yr_1 (Year 2012)                | Increases rentals by 2,026 |
| season_Winter                   | Increases rentals by 1,676 |
| atemp (Apparent Temperature)    | Increases rentals by 758   |
| weekday_Friday                  | Increases rentals by 517   |

Key Insights:  
- Adverse weather reduces demand drastically.  
- Winter and year-over-year growth drive the largest increases.  
- Weekday and temperature effects are smaller but statistically significant.  



 5. Model Interpretation
- Environmental Factors:  
  - *Temperature*: Every 1-unit increase in atemp correlates with ~758 more rentals.  
  - *Weather*: Light snow/rain reduces rentals by ~2,149 compared to clear days.  
- Temporal Trends:  
  - *Year Effect*: 2012 saw 2,026 more rentals/day than 2011.  
  - *Seasonality*: Winter contributes 1,676 more rentals vs. Spring.  

 Conclusion
The Ridge regression model provides a balance of interpretability and performance, explaining 83% of rental variance with minimal overfitting. For applications requiring maximum accuracy, the XGBoost model (R² = 0.86) is recommended despite its complexity. The analysis underscores the dominance of weather, seasonal trends, and annual growth in bike rental demand.

---

**Challenges Faced Report**



 1. Handling Outliers in Environmental Variables  
Challenge:  
Outliers were detected in hum (humidity) and windspeed columns. Extreme values could distort model performance by skewing predictions.  

Technique Used:  
- Capping Outliers: Outliers were capped at the 1st and 99th percentiles using cap_outliers().  
- Logarithmic Transformation: Applied np.log1p() to windspeed to normalize its distribution.  

Reasoning:  
- Capping retains data integrity while reducing outlier impact.  
- Log transformation stabilizes variance and mitigates skewness for better model performance.  


 2. Multicollinearity in Features  
Challenge:  
High correlation between temp and atemp (r=0.99) and between windspeed and windspeed_log (r=0.98) introduced redundancy.  

Technique Used:  
- Feature Removal: Dropped temp, windspeed, and hum after Variance Inflation Factor (VIF) analysis.  

Reasoning:  
- Multicollinearity inflates coefficient variance, reducing model interpretability. Removing redundant features improved model stability.  


 3. Temporal and Seasonal Trends  
Challenge:  
Strong seasonality (e.g., higher rentals in summer) and yearly trends required decomposition.  

Technique Used:  
- Seasonal Decomposition: Applied seasonal_decompose() to isolate trend, seasonality, and residuals.  
- Rolling Averages: Computed 7-day rolling averages to smooth daily fluctuations.  

Reasoning:  
- Decomposition clarified underlying patterns for better feature engineering.  
- Rolling averages highlighted mid-term trends for visual analysis.  


 4. Non-Normality of Target Variable  
Challenge:  
The target variable cnt (total rentals) exhibited non-normality (negative skew and low kurtosis).  

Technique Used:  
- Non-Parametric Models: Tested tree-based models alongside linear models.  
- Robust Scaling: Applied StandardScaler to continuous features (atemp, windspeed_log).  

Reasoning:  
- Tree-based models handle non-linear relationships better.  
- Scaling ensures features contribute equally during training.  


5. Categorical Feature Encoding  
Challenge:  
High-cardinality categorical columns (e.g., season, weathersit) required encoding without introducing dimensionality issues.  

Technique Used:  
- One-Hot Encoding: Applied pd.get_dummies() with drop_first=True to avoid the dummy variable trap.  

Reasoning:  
- One-hot encoding preserves categorical relationships while reducing bias compared to label encoding.


 6. Overfitting in Tree-Based Models  
Challenge: Tree-based models (e.g., Random Forest, XGBoost) showed significant overfitting.  
Technique:  
- Regularization: Tuned hyperparameters (e.g., max_depth, subsample in XGBoost).  
- Cross-Validation: Used 5-fold CV to evaluate generalization.  
Reasoning:  
- Regularization constraints reduce model complexity.  
- Cross-validation ensures stable performance across data splits. 


 7. Model Selection Trade-offs  
Challenge:  
Balancing interpretability (critical for business decisions) and predictive performance.  

Technique Used:  
- Ridge Regression: Chosen as the final model for its balance of interpretability and performance (R²=0.83, RMSE=819).  
- XGBoost: Acknowledged as the top performer (R²=0.86) but sacrificed interpretability.  

Reasoning:  
- Ridge’s coefficients provide actionable insights (e.g., weathersit_Light Snow & Rain reduces rentals by 2,149 units).  
- XGBoost was noted for scenarios prioritizing accuracy over explainability.  


 Conclusion  
The project successfully addressed challenges through a mix of robust preprocessing (outlier handling, scaling), feature engineering (aggregation, transformation), and model selection tailored to interpretability needs. Techniques like Ridge regression and seasonal decomposition ensured actionable insights, while tree-based models provided high accuracy alternatives. Future work could explore hybrid models or advanced time-series approaches (e.g., SARIMA) to further refine predictions.  

----

# Author Information

- Dhanesh B. B.  

- Contact Information:  
    - [Email](dhaneshbb5@gmail.com) 
    - [LinkedIn](https://www.linkedin.com/in/dhanesh-b-b-2a8971225/) 
    - [GitHub](https://github.com/dhaneshbb)


---

# References

**Data Description Reference:**
   - Bike Sharing Dataset, Hadi Fanaee-T, LIAAD, University of Porto. Available online: [Bike Sharing Dataset](http://capitalbikeshare.com/system-data). Accessed [current date].
   - Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, Springer Berlin Heidelberg. DOI: [10.1007/s13748-013-0040-3](http://dx.doi.org/10.1007/s13748-013-0040-3).

---

# Appendix

## About data

The dataset used in this project is drawn from Washington D.C.’s Capital Bikeshare system and spans the years 2011 and 2012. It is provided in both hourly and daily formats, allowing for a comprehensive analysis of bike rentals in relation to various temporal, environmental, and seasonal factors.

 Key Characteristics

1. Time Variables  
   - Year/Month/Day/Hour: Captures precise timestamps, which helps in understanding daily and hourly rental patterns.  
   - Holiday/Weekend Indicators: Distinguish between ordinary working days and days off, aiding in analyzing usage behavior shifts.

2. Weather Factors  
   - Temperature (temp, atemp): Normalized measures of actual and perceived temperatures.  
   - Humidity (hum): Indicates the moisture content in the air.  
   - Wind Speed (windspeed): Reflects the breeze or wind intensity.  
   - Weather Situation (weathersit): Ranges from clear skies to severe conditions (e.g., heavy rain or snowfall).

3. Rental Data  
   - Casual Users (casual): Counts of short-term or non-registered users.  
   - Registered Users (registered): Counts of regular subscribers.  
   - Total Rentals (cnt): Aggregates the casual and registered counts to represent overall demand.

---

## Source Code and Dependencies

In the development of this project, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

---

Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: builtins
2: builtins
3: pandas
4: warnings
5: researchpy
6: matplotlib.pyplot
7: missingno
8: seaborn
9: numpy
10: scipy.stats
11: textwrap
12: logging
13: time
14: statsmodels.api
15: joblib
16: psutil
17: os
18: gc
19: calendar
20: types
21: inspect

User-defined functions:
1: memory_usage
2: dataframe_memory_usage
3: garbage_collection
4: normality_test_with_skew_kurt
5: spearman_correlation_with_target
6: spearman_correlation
7: calculate_vif
8: cap_outliers
9: evaluate_regression_model
10: visualize_model_performance
11: hyperparameter_tuning

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: plot_boxplots
30: kde_batches
31: box_plot_batches
32: qq_plot_batches
33: num_vs_num_scatterplot_pair_batch
34: cat_vs_cat_pair_batch
35: num_vs_cat_box_violin_pair_batch
36: cat_bar_batches
37: cat_pie_chart_batches
38: num_analysis_and_plot
39: cat_analyze_and_plot
40: chi2_contingency
41: fisher_exact
42: pearsonr
43: spearmanr
44: ttest_ind
45: mannwhitneyu
46: linkage
47: dendrogram
48: leaves_list
49: variance_inflation_factor
50: seasonal_decompose
51: train_test_split
52: cross_val_score
53: learning_curve
54: resample
55: compute_class_weight
56: mean_absolute_error
57: mean_squared_error
58: r2_score
59: mean_absolute_percentage_error
60: mean_squared_log_error
</pre>