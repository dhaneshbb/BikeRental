
# Bike Rental Demand Analysis and Prediction

## Project Overview
This project analyzes historical bike rental data from Washington D.C.'s Capital Bikeshare system (2011–2012) to identify patterns and build predictive models for daily rental counts. The goal is to understand how environmental and seasonal factors influence demand and recommend optimal strategies for bike allocation.

---

## Dataset Overview
### Source
- **Dataset**: [Capital Bikeshare System (2011-2012)](https://d3ilbixij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PROP-1018-BikeRental.zip)


### Key Attributes
- **Temporal**: `dteday` (date), `season`, `mnth`, `weekday`, `hr` (hourly data only).  
- **Weather**: `temp` (normalized temperature), `hum` (humidity), `windspeed`, `weathersit` (weather condition).  
- **Usage Metrics**: `casual` (non-registered users), `registered`, `cnt` (total rentals).  

### Files
- **Daily Data**: `day.csv` (731 records, 16 features).  
- **Hourly Data**: `hour.csv` (17,379 records, 17 features).  

---

## Project Structure
```plaintext
bike-rental-analysis/
├── data/
│   ├── 1.1 raw/                # Raw datasets (hour.csv, day.csv)
│   └── 1.2 processed/          # Processed data splits (train/test)
├── docs/                       # Project documentation and datasets
├── notebooks/
│   └── PRCP-1018-BikeRental.ipynb  # Main Jupyter notebook for analysis
├── reports/
│   └── Final Report.md         # Detailed analysis and results
├── results/
│   ├── models/                 # Saved models (e.g., Ridge regression)
│   └── figures/                # Visualizations and EDA outputs
└── scripts/                    # Utility and preprocessing scripts
```

---

## Tasks and Objectives
1. **Data Analysis Report**:  
   - Exploratory Data Analysis (EDA) to identify trends, outliers, and correlations.  
   - Seasonal and weather impact assessment on rental demand.  

2. **Predictive Modeling**:  
   - Regression models to forecast daily bike rentals (`cnt`).  
   - Comparison of linear models (Ridge, Lasso) and tree-based models (XGBoost, Gradient Boosting).  

3. **Challenges Report**:  
   - Solutions for multicollinearity, outliers, and non-normality in data.  

---

## Key Methodologies
### Data Preprocessing
- **Outlier Handling**: Capping extreme values in `hum` and `windspeed` at 1st/99th percentiles.  
- **Feature Engineering**:  
  - Log transformation for `windspeed`.  
  - One-hot encoding for categorical variables (`season`, `weathersit`).  
- **Multicollinearity Mitigation**: Removed redundant features (e.g., `temp` vs. `atemp`).  

### Model Development
- **Algorithms Tested**:  
  - Linear Regression, Ridge/Lasso Regression.  
  - XGBoost, Gradient Boosting, Random Forest.  
- **Evaluation Metrics**: R², RMSE, MAE, and cross-validation.  

---

## Results and Model Comparison
### Top Performers
| Model          | R² (Test) | RMSE   | Interpretability |  
|----------------|-----------|--------|------------------|  
| **Ridge**      | 0.832     | 819.62 | High             |  
| **XGBoost**    | 0.860     | 748.11 | Low              |  

### Key Insights
1. **Demand Drivers**:  
   - Temperature (`atemp`) and clear weather increase rentals.  
   - Adverse weather (e.g., snow) reduces demand by ~2,149 rentals/day.  
2. **Seasonality**:  
   - Peak demand in fall/winter (6,000+ rentals/day).  
   - 50% YoY growth from 2011 to 2012.  

---

## Challenges and Solutions
| Challenge                  | Solution                              |  
|----------------------------|---------------------------------------|  
| Multicollinearity          | Removed redundant features (VIF analysis). |  
| Outliers in `windspeed`    | Log transformation and capping.       |  
| Non-normal target (`cnt`)  | Tree-based models + Robust scaling.   |  

---

## Installation
### Dependencies
```bash
pip install -r requirements.txt  
```

### Custom Library
The project uses [`insightfulpy`](https://github.com/dhaneshbb/insightfulpy) for streamlined preprocessing:  
```bash
pip install insightfulpy
```

---

## Usage
1. Run the Jupyter notebook `notebooks/PRCP-1018-BikeRental.ipynb`.  
2. Execute cells sequentially to reproduce EDA, modeling, and reports.  

---

## References
- Fanaee-T, H., & Gama, J. (2013). [Event Labeling Combining Ensemble Detectors and Background Knowledge](https://doi.org/10.1007/s13748-013-0040-3). *Progress in Artificial Intelligence*.  
- Dataset Citation: [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).  

---

📌 **Note**: Full code, visualizations, and model artifacts are available in the repository. For detailed implementation, refer to the [Final Report](reports/Final%20Report.md).  
