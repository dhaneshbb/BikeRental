import pandas as pd
day = pd.read_csv(r'D:\datamites\BikeRental\data\1.1 raw\day.csv')
hr = pd.read_csv(r'D:\datamites\BikeRental\data\1.1 raw\hour.csv')

from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

from insightfulpy.eda import *

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, pearsonr, spearmanr,
    ttest_ind, mannwhitneyu, shapiro
)
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, mean_squared_log_error
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA

import joblib

import psutil
import os
import gc

def memory_usage():
    """Prints the current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def dataframe_memory_usage(df):
    """Returns the memory usage of a Pandas DataFrame in MB."""
    mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"DataFrame Memory Usage: {mem_usage:.2f} MB")
    return mem_usage

def garbage_collection():
    """Performs garbage collection to free up memory."""
    gc.collect()
    memory_usage()

def normality_test_with_skew_kurt(df):
    normal_cols = []
    not_normal_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) >= 3:
            if len(col_data) <= 5000:
                stat, p_value = shapiro(col_data)
                test_used = 'Shapiro-Wilk'
            else:
                stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_used = 'Kolmogorov-Smirnov'
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            result = {
                'Column': col,
                'Test': test_used,
                'Statistic': stat,
                'p_value': p_value,
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
            if p_value > 0.05:
                normal_cols.append(result)
            else:
                not_normal_cols.append(result)
    normal_df = (
        pd.DataFrame(normal_cols)
        .sort_values(by='Column') 
        if normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols)
        .sort_values(by='p_value', ascending=False)  # Sort by p-value descending (near normal to not normal)
        if not_normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p ≤ 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df
def spearman_correlation_with_target(data, non_normal_cols, target_col='TARGET', plot=True, table=True):
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric. Please encode it before running this test.")
    correlation_results = {}
    for col in non_normal_cols:
        if col not in data.columns:
            continue 
        coef, p_value = spearmanr(data[col], data[target_col], nan_policy='omit')
        correlation_results[col] = {'Spearman Coefficient': coef, 'p-value': p_value}
    correlation_data = pd.DataFrame(correlation_results).T.dropna()
    correlation_data = correlation_data.sort_values('Spearman Coefficient', ascending=False)
    if target_col in correlation_data.index:
        correlation_data = correlation_data.drop(target_col)
    positive_corr = correlation_data[correlation_data['Spearman Coefficient'] > 0]
    negative_corr = correlation_data[correlation_data['Spearman Coefficient'] < 0]
    if table:
        print(f"\nPositive Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in positive_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
        print(f"\nNegative Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in negative_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
    if plot:
        plt.figure(figsize=(20, 8))  # Increase figure width to prevent label overlap
        sns.barplot(x=correlation_data.index, y='Spearman Coefficient', data=correlation_data, palette='coolwarm')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"Spearman Correlation with Target ('{target_col}')", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Spearman Coefficient", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels for clarity
        plt.subplots_adjust(bottom=0.3)  # Add space below the plot for labels
        plt.tight_layout()
        plt.show()
    return correlation_data
def spearman_correlation(data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8):
    if non_normal_cols.empty:
        print("\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation.")
        return
    selected_columns = non_normal_cols['Column'].tolist()
    if exclude_target and exclude_target in selected_columns and pd.api.types.is_numeric_dtype(data[exclude_target]):
        selected_columns.remove(exclude_target)
    spearman_corr_matrix = data[selected_columns].corr(method='spearman')
    multicollinear_pairs = []
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i+1:]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    print("\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(multicollinearity_threshold))
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Keep reasonable bounds
    annot_font_size = max(min(10, 200 / num_vars), 6)  # Smaller font for more variables
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()
def calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=[exclude_target], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_data.values, i) 
                       for i in range(numeric_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data['VIF'] > multicollinearity_threshold]
    low_vif = vif_data[vif_data['VIF'] <= multicollinearity_threshold]
    print(f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):")
    print("\nFeatures with VIF > threshold (High Multicollinearity):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("None. No features exceed the VIF threshold.")
    print("\nFeatures with VIF <= threshold (Low/No Multicollinearity):")
    if not low_vif.empty:
        print(low_vif.to_string(index=False))
    else:
        print("None. All features exceed the VIF threshold.")
    return vif_data, high_vif['Feature'].tolist()

if __name__ == "__main__":
    memory_usage()

dataframe_memory_usage(day)

dataframe_memory_usage(hr)

dataframes = {
    'day': day,
    'hr': hr,
}

linked_key(dataframes)

print(day.shape)
for idx, col in enumerate(day.columns):
        print(f"{idx}: {col}")

day.head().T

detect_mixed_data_types(day)

cat_high_cardinality(day)

missing_inf_values(day)
print(f"\nNumber of duplicate rows: {day.duplicated().sum()}\n")
duplicates = day[day.duplicated()]
duplicates

inf_counts = np.isinf(day.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

day.dtypes.value_counts()

columns_info("dayset Overview", day)

analyze_data(day)

missing_inf_values(day)
show_missing(day)
print(f"\nNumber of duplicate rows: {day.duplicated().sum()}\n")
duplicates = day[day.duplicated()]
duplicates

for idx, col in enumerate(hr.columns):
        print(f"{idx}: {col}")

hr.head().T

detect_mixed_data_types(hr)

cat_high_cardinality(hr)

missing_inf_values(hr)
print(f"\nNumber of duplicate rows: {hr.duplicated().sum()}\n")
duplicates = hr[hr.duplicated()]
duplicates

inf_counts = np.isinf(hr.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

hr.dtypes.value_counts()

columns_info("hrset Overview", hr)

analyze_data(hr)

missing_inf_values(hr)
show_missing(hr)
print(f"\nNumber of duplicate rows: {hr.duplicated().sum()}\n")
duplicates = hr[hr.duplicated()]
duplicates

hr['dteday'] = pd.to_datetime(hr['dteday'])
day['dteday'] = pd.to_datetime(day['dteday'])
daily_aggregations = {
    'cnt': 'sum', 
    'casual': 'sum', 
    'registered': 'sum', 
    'temp': 'mean',  
    'atemp': 'mean',  
    'hum': 'mean',  
    'windspeed': 'mean',  
    'weathersit': lambda data: data.mode()[0] if not data.mode().empty else None  
}
aggregated_hourly_data = hr.groupby('dteday').agg(daily_aggregations).reset_index()
columns_to_use = aggregated_hourly_data.columns.difference(day.columns).tolist()
columns_to_use.append('dteday')
data = pd.merge(day, aggregated_hourly_data[columns_to_use], on='dteday', how='left')

print("Hourly Data Info:")
print(hr.info())
print("\nDaily Data Info:")
print(day.info())
print("\nMerged Data Info:")
print(data.info())
print("\nDuplicate Check:")
print("Hourly Data duplicates:", hr.duplicated().sum())
print("Daily Data duplicates:", day.duplicated().sum())
print("Merged Data duplicates:", data.duplicated().sum())
print("\nUnique Days Check:")
print("Unique days in hourly data:", hr['dteday'].nunique())
print("Unique days in daily data:", day['dteday'].nunique())
print("Unique days in merged data:", data['dteday'].nunique())
expected_columns = ['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
                    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                    'casual', 'registered', 'cnt']
print("\nColumns in Merged Data:")
print(data.columns.tolist())
print("All expected columns in merged data:", all(col in data.columns for col in expected_columns))
print("\nSummary Statistics for Merged Data:")
print(data.describe())
print("\nData Anomalies Check:")
print("Negative or zero counts in merged data:")
print(data[data['cnt'] <= 0][['dteday', 'cnt']])

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

detect_mixed_data_types(data)

cat_high_cardinality(data)

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates

inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

data.dtypes.value_counts()

columns_info("Dataset Overview", data)

analyze_data(data)

categorical_columns = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
for column in categorical_columns:
    data[column] = data[column].astype('category')

data.drop(columns=['instant'], inplace=True)

data.drop(columns=['casual', 'registered'], inplace=True)

data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
data_missing_summary, data_non_missing_summary = comp_num_analysis(data, missing_df=True)
data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_cat_missing_summary.shape)
print(data_missing_summary.shape)
print(data_outlier_summary.shape)

data_outlier_summary

plot_boxplots(data)
calculate_skewness_kurtosis(data)

outlier_cols = ["hum", "windspeed"]
interconnected_outliers_df = interconnected_outliers(data, outlier_cols)

interconnected_outliers_df

def cap_outliers(series, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(series, lower_percentile)
    upper_bound = np.percentile(series, upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)
data['hum'] = cap_outliers(data['hum'])
data['windspeed'] = cap_outliers(data['windspeed'])

data['windspeed_log'] = np.log1p(data['windspeed'])

Q1 = data['windspeed_log'].quantile(0.25)
Q3 = data['windspeed_log'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['windspeed_log'] < lower_bound) | (data['windspeed_log'] > upper_bound)]
print(f"Outliers found: {outliers.shape[0]}")

median_value = data['windspeed_log'].median()
data.loc[data['windspeed_log'] > upper_bound, 'windspeed_log'] = median_value
data.loc[data['windspeed_log'] < lower_bound, 'windspeed_log'] = median_value

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)
spearman_correlation(data, data_not_normal_df, exclude_target='cnt', multicollinearity_threshold=0.8)

num_analysis_and_plot(data, 'cnt')

columns_info("Dataset Overview", data)

analyze_data(data)

num_summary(data)

cat_summary(data)

kde_batches(data, batch_num=1)
box_plot_batches(data, batch_num=1)
qq_plot_batches(data, batch_num=1)

cat_bar_batches(data, batch_num=1,high_cardinality_limit=22)
cat_pie_chart_batches(data, batch_num=1,high_cardinality_limit=22)

num_vs_num_scatterplot_pair_batch(data,pair_num=4, batch_num=1, hue_column="cnt")

num_vs_cat_box_violin_pair_batch(data, pair_num=4, batch_num=1, high_cardinality_limit=22)

cat_vs_cat_pair_batch(data, pair_num=6, batch_num=1, high_cardinality_limit=22)

data['dteday'] = pd.to_datetime(data['dteday'])
data['cnt_rolling'] = data['cnt'].rolling(window=7).mean()
decomposition = seasonal_decompose(data.set_index('dteday')['cnt'], model='additive', period=30)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].plot(data['dteday'], data['cnt'], color='blue', linewidth=1.5)
axes[0, 0].set_title("Bike Rentals Over Time")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Total Rentals")
axes[0, 0].grid(True)
axes[0, 1].plot(data['dteday'], data['cnt_rolling'], color='red', linewidth=2)
axes[0, 1].set_title("7-Day Rolling Average of Bike Rentals")
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Smoothed Rentals")
axes[0, 1].grid(True)
sns.boxplot(x=data['mnth'], y=data['cnt'], ax=axes[1, 0], palette="coolwarm")
axes[1, 0].set_title("Monthly Bike Rental Distribution")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Total Rentals")
decomposition.trend.plot(ax=axes[1, 1], title="Trend (Seasonal Decomposition)", color="green")
plt.tight_layout()
plt.show()
decomposition = seasonal_decompose(data.set_index('dteday')['cnt'], model='additive', period=30)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
decomposition.trend.plot(ax=axes[0], title='Trend')
decomposition.seasonal.plot(ax=axes[1], title='Seasonality')
decomposition.resid.plot(ax=axes[2], title='Residuals')
plt.show()

data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
data_missing_summary, data_non_missing_summary = comp_num_analysis(data, missing_df=True)
data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_cat_missing_summary.shape)
print(data_missing_summary.shape)
print(data_outlier_summary.shape)

data_missing_summary

data_outlier_summary

upper_limit_windspeed = data['windspeed'].quantile(0.75) + 1.5 * (data['windspeed'].quantile(0.75) - data['windspeed'].quantile(0.25))
data['windspeed'] = data['windspeed'].apply(lambda x: upper_limit_windspeed if x > upper_limit_windspeed else x)

data.drop('cnt_rolling', axis=1, inplace=True)

data.drop('dteday', axis=1, inplace=True)

import calendar
data['mnth'] = data['mnth'].apply(lambda x: calendar.month_abbr[x])
data['season'] = data['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
data['weathersit'] = data['weathersit'].map({
    1: 'Clear', 2: 'Mist & Cloudy', 3: 'Light Snow & Rain', 4: 'Heavy Snow & Rain'
})
data['weekday'] = data['weekday'].map({
    0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"
})
categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

bool_cols = data_encoded.select_dtypes(include='bool').columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data_encoded)
spearman_correlation(data_encoded, data_not_normal_df, exclude_target='cnt', multicollinearity_threshold=0.8)

above_threshold, below_threshold = calculate_vif(data_encoded, exclude_target='cnt', multicollinearity_threshold=8.0)

data_encoded.drop('windspeed', axis=1, inplace=True)

data_encoded.drop('temp', axis=1, inplace=True)

if 'holiday_1' in data_encoded.columns and 'workingday_1' in data_encoded.columns:
    data_encoded = data_encoded.drop(['workingday_1'], axis=1)

data_encoded.drop('hum', axis=1, inplace=True)

above_threshold1, below_threshold1 = calculate_vif(data_encoded, exclude_target='cnt', multicollinearity_threshold=8.0)

X = data_encoded.drop('cnt', axis=1)
y = data_encoded['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

continuous_features = ['atemp', 'windspeed_log']
scaler = StandardScaler()
X_train_cont_scaled = scaler.fit_transform(X_train[continuous_features])
X_test_cont_scaled = scaler.transform(X_test[continuous_features])
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_features] = X_train_cont_scaled
X_test_scaled[continuous_features] = X_test_cont_scaled
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time 
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    msle = mean_squared_log_error(y_test, y_pred) if np.all(y_pred >= 0) else None
    mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(y_pred >= 0) else None
    r2_train = r2_score(y_train, y_pred_train)
    cv_r2 = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="r2"))
    overfit = r2_train - r2
    return {
        "Model Name": type(model).__name__,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2,
        "MSLE": msle,
        "MAPE": mape,
        "Cross-Validation R²": cv_r2,
        "Training R²": r2_train,
        "Overfit": overfit,
        "Training Time (seconds)": round(training_time, 4)
    }
    
def visualize_model_performance(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Model Performance: {type(model).__name__}", fontsize=14)
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring="r2")
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    axes[0, 0].plot(train_sizes, train_mean, 'o-', label="Train Score")
    axes[0, 0].plot(train_sizes, test_mean, 'o-', label="Test Score")
    axes[0, 0].set_title("Learning Curve")
    axes[0, 0].set_xlabel("Training Samples")
    axes[0, 0].set_ylabel("R² Score")
    axes[0, 0].legend()
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, color="blue")
    axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--r")
    axes[0, 1].set_title("True vs Predicted (Test)")
    axes[0, 1].set_xlabel("True Values")
    axes[0, 1].set_ylabel("Predicted Values")
    axes[0, 2].scatter(y_pred_test, residuals_test, alpha=0.5, color="purple")
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title("Residuals vs Predicted (Test)")
    axes[0, 2].set_xlabel("Predicted Values")
    axes[0, 2].set_ylabel("Residuals")
    sns.histplot(residuals_test, bins=30, kde=True, ax=axes[1, 0], color="teal")
    axes[1, 0].set_title("Test Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals")
    sns.histplot(residuals_train, bins=30, kde=True, ax=axes[1, 1], color="green", alpha=0.7)
    axes[1, 1].set_title("Train Residuals Distribution")
    axes[1, 1].set_xlabel("Residuals")
    stats.probplot(residuals_test, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title("QQ Plot (Test Residuals)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def hyperparameter_tuning(models, param_grids, X_train, y_train, scoring_metric='neg_mean_squared_error', cv_folds=5):
    best_models = {}
    best_params = {}
    execution_times = {}
    for model_name, model in models.items():
        print(f"Starting grid search for {model_name}...")
        start_time = time.time()
        if model_name in param_grids:
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param_grids[model_name],
                                       scoring=scoring_metric,
                                       cv=cv_folds,
                                       verbose=1,
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            best_params[model_name] = grid_search.best_params_
            execution_times[model_name] = time.time() - start_time
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Completed grid search for {model_name} in {execution_times[model_name]:.2f} seconds.\n")
        else:
            print(f"No parameter grid available for {model_name}.")
    return best_models, best_params, execution_times

X_train_ols = sm.add_constant(X_train_scaled)
X_test_ols = sm.add_constant(X_test_scaled)
ols_model = sm.OLS(y_train, X_train_ols).fit()
print(ols_model.summary())
y_pred = ols_model.predict(X_test_ols)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

base_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = []
for model_name, model in base_models.items():
    result = evaluate_regression_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    results.append(result)
base_results = pd.DataFrame(results)

base_results

param_grids = {
    "Lasso": {
        "alpha": np.logspace(-4, 1, 10)
    },
    "ElasticNet": {
        "alpha": np.logspace(-4, 1, 10),
        "l1_ratio": np.linspace(0.1, 1, 10)
    },
    "Ridge": {
        "alpha": np.logspace(-3, 3, 10)
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0]
    }
}
tune_models = {
    "Lasso": Lasso(max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(max_iter=10000, random_state=42),
    "Ridge": Ridge(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "XGBRegressor": XGBRegressor(random_state=42)
}
best_models, best_params, execution_times = hyperparameter_tuning(tune_models, param_grids, X_train_scaled, y_train)

for model_name, params in best_params.items():
    print(f"Best parameters for {model_name}: {params}")

results = []
for model_name, model in best_models.items():
    result = evaluate_regression_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    result["Model Name"] = model_name 
    results.append(result)
tuned_results = pd.DataFrame(results)

tuned_results

base_results

tuned_results

final_model = Ridge(alpha=0.46415888336127775, random_state=42)
final_model.fit(X_train_scaled, y_train)

y_pred_final = final_model.predict(X_test_scaled)
mse_final = mean_squared_error(y_test, y_pred_final)
rmse_final = mse_final ** 0.5
r2_final = r2_score(y_test, y_pred_final)

print("Final Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_final:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_final:.2f}")
print(f"R-squared (R²): {r2_final:.3f}")

cv_folds = 5
cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=cv, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores):.4f}, Standard Deviation: {np.std(cv_scores):.4f}")

plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores, vert=True, patch_artist=True)
plt.title("Cross-Validation R² Scores")
plt.ylabel("R² Score")
plt.xticks([1], ["Ridge"])
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

coefficients = final_model.coef_
feature_importance_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': coefficients
})
feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()
feature_importance_df.sort_values('Absolute Coefficient', ascending=False, inplace=True)
print(feature_importance_df)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Coefficient'])
plt.title('Feature Importance (Ridge Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.gca().invert_yaxis() 
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

visualize_model_performance(final_model, X_train_scaled, y_train, X_test_scaled, y_test)

joblib.dump(final_model, 'final_ridge_model.joblib')

loaded_model = joblib.load('final_ridge_model.joblib')
y_pred_test = loaded_model.predict(X_test_scaled)
print("Sample prediction:", y_pred_test[:5])

import types
import inspect
user_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ == '__main__']
imported_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ != '__main__']
imported_pkgs = [name for name in globals() if isinstance(globals()[name], types.ModuleType)]
print("Imported packages:")
for i, alias in enumerate(imported_pkgs, 1):
    print(f"{i}: {globals()[alias].__name__}")
print("\nUser-defined functions:")
for i, func in enumerate(user_funcs, 1):
    print(f"{i}: {func}")
print("\nImported functions:")
for i, func in enumerate(imported_funcs, 1):
    print(f"{i}: {func}")