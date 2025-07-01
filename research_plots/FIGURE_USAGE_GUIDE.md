# Figure Usage Guide for Research Paper

## How to Use Each Figure in Your Research Paper

### 📊 **Data Analysis Section**

#### Figure 1: Target Variables Distribution
- **File**: `01_data_analysis/Figure_1_Target_Variables_Distribution.png`
- **Usage**: Introduction/Data Description section
- **Caption**: "Distribution of target variables in the Makassar weather dataset. (a) Rainfall shows zero-inflated distribution typical of tropical precipitation, (b) Sunshine duration exhibits bimodal pattern, (c) Temperature shows normal distribution around 27°C, (d) Wind direction demonstrates seasonal patterns, (e) Wind speed follows log-normal distribution."
- **Purpose**: Show data characteristics and justify modeling approach

#### Figure 2: Feature Correlation Matrix  
- **File**: `01_data_analysis/Figure_2_Feature_Correlation_Matrix.png`
- **Usage**: Methodology/Feature Engineering section
- **Caption**: "Correlation matrix between selected features and target variables. Strong correlations (|r| > 0.7) are observed between temperature-related features and moderate correlations between temporal features and weather variables."
- **Purpose**: Justify feature selection and show multicollinearity

#### Figure 3: Feature Distributions
- **File**: `01_data_analysis/Figure_3_Feature_Distributions.png`  
- **Usage**: Data Analysis section
- **Caption**: "Distribution of top 6 most important features and all target variables. Features show diverse distributions requiring robust modeling approaches."
- **Purpose**: Demonstrate data complexity and preprocessing needs

---

### 🎯 **Methodology Section**

#### Figure 4: Seasonal Patterns
- **File**: `06_seasonal_patterns/Figure_4_Seasonal_Patterns.png`
- **Usage**: Data Analysis/Seasonal Analysis subsection
- **Caption**: "Monthly seasonal patterns for all target variables. Clear monsoon patterns are evident in rainfall (peaks in Dec-Feb) and sunshine duration (inverse relationship with rainfall)."
- **Purpose**: Justify inclusion of temporal features

#### Figure 5: Multi-Target Feature Importance
- **File**: `03_feature_importance/Figure_5_Multi_Target_Feature_Importance.png`
- **Usage**: Results/Feature Analysis section
- **Caption**: "Top 5 most important features for each target variable based on GBM feature importance. Temperature-related features and humidity consistently rank highest across all targets."
- **Purpose**: Show which features drive predictions

---

### 📈 **Results Section**

#### Figures 6a-6e: Individual Model Performance
- **Files**: `02_model_performance/Figure_6a-6e_*.png`
- **Usage**: Results/Model Performance section
- **Caption**: "Prediction performance for individual target variables on test set. (a) Rainfall (R²=0.901), (b) Sunshine duration (R²=0.982), (c) Temperature (R²=0.923), (d) Wind direction (R²=0.978), (e) Wind speed (R²=0.946). All models demonstrate excellent agreement between predicted and observed values."
- **Purpose**: Show model accuracy for each target

#### Figures 7a-7e: Multi-Day Forecasting
- **Files**: `04_multi_day_forecasting/Figure_7a-7e_*.png`
- **Usage**: Results/Multi-Day Forecasting section  
- **Caption**: "Multi-day forecasting performance (0-5 days ahead) for all target variables. Performance degrades gracefully with forecast horizon, with temperature and wind speed showing most stability."
- **Purpose**: Demonstrate multi-day capability and degradation patterns

---

### 📊 **Comparative Analysis Section**

#### Figure 8: Performance Heatmap
- **File**: `05_comparative_analysis/Figure_8_Multi_Day_Performance_Heatmap.png`
- **Usage**: Results/Comparative Analysis section
- **Caption**: "Heatmap of R² scores across all target variables and forecast horizons. Sunshine duration and wind direction maintain highest accuracy across all forecast days."
- **Purpose**: Compare performance across targets and days

#### Figure 9: Forecast Horizon Analysis
- **File**: `05_comparative_analysis/Figure_9_Performance_vs_Forecast_Horizon.png`
- **Usage**: Discussion/Forecast Limitations section
- **Caption**: "Prediction accuracy (R²) versus forecast horizon for all target variables. Rainfall shows steepest degradation while temperature predictions remain most stable."
- **Purpose**: Analyze forecast limitations and variable-specific behavior

---

## 📝 **Supplementary Material**

### Figures S1-S4: Detailed Day-Specific Predictions
- **Files**: `04_multi_day_forecasting/supplementary/Figure_S*_*.png`
- **Usage**: Supplementary material or appendix
- **Caption**: "Detailed prediction results for selected forecast horizons: Today, 1-day, 3-day, and 5-day ahead predictions for (S1-S4) rainfall, (S5-S8) sunshine, and (S9-S12) temperature."
- **Purpose**: Provide detailed view of specific forecast performance

---

## 📋 **Table Suggestions**

### Table 1: Model Performance Summary
```
| Target Variable | RMSE | MAE | R² | Performance Level |
|-----------------|------|-----|----|--------------------|
| Rainfall (mm) | 6.81 | 1.88 | 0.901 | Excellent |
| Sunshine (hours) | 0.44 | 0.32 | 0.982 | Outstanding |
| Temperature (°C) | 0.32 | 0.20 | 0.923 | Excellent |
| Wind Direction (°) | 13.29 | 4.89 | 0.978 | Outstanding |
| Wind Speed (m/s) | 0.15 | 0.05 | 0.946 | Excellent |
```

### Table 2: Feature Importance Rankings
- Extract from Figure 5 - show top 5 features for each target
- Include importance scores and brief descriptions

### Table 3: Multi-Day Performance Degradation  
- Show R² values for each target at Days 0, 1, 3, 5
- Calculate degradation percentages

---

## 🎨 **Figure Quality Notes**

### For Publication:
- All figures are high-resolution PNG (300 DPI equivalent)
- Clean, professional styling with clear labels
- Consistent color schemes across related figures
- Readable fonts and appropriate sizing

### Editing Suggestions:
- Consider converting to EPS/PDF for journal submission
- May need to adjust font sizes for specific journal requirements
- Color figures work best; grayscale versions available on request

---

## 📖 **Writing Tips**

### Results Section Structure:
1. **Data Overview** (Figures 1, 3, 4)
2. **Feature Analysis** (Figures 2, 5)  
3. **Model Performance** (Figures 6a-6e, Table 1)
4. **Multi-Day Forecasting** (Figures 7a-7e, 8, 9)
5. **Comparative Analysis** (Figure 8, 9, Tables 2-3)

### Key Points to Emphasize:
- **High R² scores** (all > 0.90) demonstrate excellent model performance
- **Multi-target capability** - single framework predicts 5 variables
- **Practical forecasting** - useful 3-5 day forecast horizon
- **Feature insights** - identify key predictive variables
- **Comprehensive evaluation** - multiple metrics and visualizations

### Statistical Significance:
- All metrics files contain exact values for reporting
- Cross-validation ensures robust performance estimates
- Time series split preserves temporal validity

---

*This guide helps you effectively use all research plots in your paper. Each figure serves a specific purpose in telling your research story.* 