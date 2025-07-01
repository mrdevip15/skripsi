# Research Plots for Weather Prediction Using Gradient Boosting Machine (GBM)

This folder contains organized plots for the research paper on multi-target weather prediction using GBM.

## Folder Structure:

### 01_data_analysis/
- **Figure_1_Target_Variables_Distribution.png**: Distribution of all target variables (rainfall, sunshine, temperature, wind direction, wind speed)
- **Figure_2_Feature_Correlation_Matrix.png**: Correlation matrix showing relationships between features and targets
- **Figure_3_Feature_Distributions.png**: Distribution plots of the most important features

### 02_model_performance/
- **Figure_6a-6e**: Individual model performance for each target variable
  - 6a: Rainfall prediction performance
  - 6b: Sunshine duration prediction performance  
  - 6c: Temperature prediction performance
  - 6d: Wind direction prediction performance
  - 6e: Wind speed prediction performance

### 03_feature_importance/
- **Figure_5_Multi_Target_Feature_Importance.png**: Top 5 most important features for each target variable

### 04_multi_day_forecasting/
- **Figure_7a-7e**: Multi-day forecasting results for each target variable (0-5 days ahead)
- **supplementary/**: Individual day prediction plots for selected forecast horizons

### 05_comparative_analysis/
- **Figure_8_Multi_Day_Performance_Heatmap.png**: Heatmap showing R² scores across all targets and forecast days
- **Figure_9_Performance_vs_Forecast_Horizon.png**: Line plot showing how prediction accuracy degrades with forecast horizon

### 06_seasonal_patterns/
- **Figure_4_Seasonal_Patterns.png**: Monthly patterns for all target variables

### metrics/
- Text files containing detailed performance metrics for each target variable

## Usage Notes:
- All figures are high-resolution PNG files suitable for publication
- Figure numbers are suggested and can be adjusted based on paper structure
- Supplementary figures provide additional detail for specific forecast horizons
- Metrics files contain exact numerical values for reporting in the paper

## Model Configuration:
- Multi-target Gradient Boosting Machine
- 5-day forecast horizon
- Features: Top meteorological variables with temporal and seasonal components
- Evaluation: R², RMSE, MAE metrics
- Cross-validation: Time series split to preserve temporal order

Generated on: 2025-06-30 07:34:16
