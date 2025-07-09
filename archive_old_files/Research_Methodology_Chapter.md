# Research Methodology

## Abstract
This chapter presents the comprehensive research methodology employed in developing and evaluating the IMAKAS (Indonesian Multi-target Automated Klimatologi Analysis System) weather prediction algorithm. The methodology encompasses data collection and preprocessing, feature engineering, model development, validation strategies, and performance evaluation metrics. The research follows a systematic approach combining machine learning techniques with meteorological domain knowledge to create an explainable and accurate weather forecasting system.

## 3.1 Research Framework and Approach

### 3.1.1 Research Design
This study employs a quantitative research approach utilizing machine learning techniques for meteorological prediction. The research design follows a systematic methodology framework consisting of five main phases:

1. **Data Collection and Preprocessing Phase**: Acquisition and preparation of high-frequency meteorological data
2. **Feature Engineering Phase**: Development of physics-informed features from raw meteorological variables
3. **Model Development Phase**: Implementation of multi-target gradient boosting models with target-specific optimization
4. **Validation and Evaluation Phase**: Comprehensive performance assessment using multiple metrics and validation strategies
5. **Interpretability Analysis Phase**: Application of explainable AI techniques for model understanding

### 3.1.2 Research Objectives
The primary objectives of this research are:

1. **Develop a multi-target weather prediction system** capable of simultaneously forecasting five key meteorological variables
2. **Implement multi-horizon forecasting** with systematic accuracy analysis across different time scales
3. **Integrate explainable AI techniques** to provide interpretable predictions for operational use
4. **Validate performance** against established baseline methods and meteorological standards
5. **Demonstrate practical applicability** for operational weather forecasting in tropical climates

### 3.1.3 Methodological Innovation
The research introduces several methodological innovations:

- **Multi-target architecture** with target-specific hyperparameter optimization
- **Physics-informed feature engineering** incorporating meteorological domain knowledge
- **Cyclical temporal encoding** for effective representation of periodic atmospheric patterns
- **Comprehensive explainability analysis** using SHAP (SHapley Additive exPlanations)
- **Systematic multi-horizon validation** with predictability decay analysis

## 3.2 Data Collection and Dataset Description

### 3.2.1 Data Source and Location
The research utilizes high-frequency meteorological data collected from the IMAKAS weather station network in Makassar, South Sulawesi, Indonesia. The selection of this location is based on:

1. **Tropical Climate Representation**: Makassar represents typical tropical maritime climate conditions
2. **Data Quality and Availability**: Consistent high-frequency measurements with minimal data gaps
3. **Operational Relevance**: Strategic location for Indonesian meteorological services
4. **Seasonal Variability**: Clear wet and dry season patterns for robust model validation

### 3.2.2 Dataset Characteristics

**Table 1: Dataset Specifications**

| Characteristic | Specification |
|---------------|---------------|
| **Data Source** | IMAKAS7.csv weather station data |
| **Temporal Coverage** | March 2023 - May 2025 |
| **Total Records** | 204,073 observations |
| **Temporal Resolution** | ~30-minute intervals |
| **Geographic Location** | Makassar, South Sulawesi, Indonesia |
| **Coordinate System** | WGS84 |
| **Data Format** | CSV with timestamp and meteorological variables |

### 3.2.3 Raw Meteorological Variables

**Table 2: Raw Input Variables**

| Variable | Description | Unit | Measurement Range | Data Type |
|----------|-------------|------|------------------|-----------|
| `Date` | Observation date | YYYY-MM-DD | 2023-03-01 to 2025-05-31 | DateTime |
| `Time` | Observation time | HH:MM:SS | 00:00:00 to 23:59:59 | DateTime |
| `Temperature_C` | Air temperature | °C | 22.5 - 35.8 | Float |
| `Dew_Point_C` | Dew point temperature | °C | 18.2 - 28.4 | Float |
| `Humidity_%` | Relative humidity | % | 45.2 - 98.7 | Float |
| `Pressure_hPa` | Atmospheric pressure | hPa | 1008.2 - 1016.8 | Float |
| `Speed_kmh` | Wind speed | km/h | 0.0 - 28.6 | Float |
| `Gust_kmh` | Wind gust speed | km/h | 0.0 - 45.2 | Float |
| `Precip_Rate_mm` | Precipitation rate | mm | 0.0 - 25.4 | Float |
| `Precip_Accum_mm` | Accumulated precipitation | mm | 0.0 - 156.8 | Float |
| `UV` | UV index | - | 0.0 - 12.8 | Float |
| `Solar_w/m2` | Solar radiation | W/m² | 0.0 - 1234.5 | Float |

### 3.2.4 Target Variables Definition

The research focuses on predicting five key meteorological variables:

**Table 3: Target Variables**

| Variable | Indonesian Name | Physical Significance | Prediction Importance |
|----------|----------------|----------------------|----------------------|
| `Humidity_%` | Kelembaban (%) | Atmospheric moisture content | High - affects comfort, agriculture |
| `Temperature_C` | Suhu (°C) | Air temperature | High - fundamental weather parameter |
| `Precip_Rate_mm` | Laju Curah Hujan (mm) | Precipitation intensity | Critical - flood/drought management |
| `Precip_Accum_mm` | Akumulasi Curah Hujan (mm) | Cumulative precipitation | Critical - water resource planning |
| `Solar_w/m2` | Radiasi Matahari (W/m²) | Solar energy flux | High - renewable energy applications |

## 3.3 Data Preprocessing and Quality Control

### 3.3.1 Data Quality Assessment

**Missing Data Analysis:**
```python
# Missing data assessment methodology
missing_data_analysis = {
    'total_records': 204073,
    'missing_percentage_threshold': 5.0,
    'essential_variables': ['Temperature_C', 'Humidity_%', 'Pressure_hPa', 'Dew_Point_C'],
    'optional_variables': ['Speed_kmh', 'Gust_kmh', 'Precip_Rate_mm', 'Precip_Accum_mm', 'UV', 'Solar_w/m2']
}
```

**Data Quality Criteria:**
1. **Completeness**: Less than 5% missing data for essential variables
2. **Consistency**: Logical relationships between related variables (e.g., dew point ≤ temperature)
3. **Accuracy**: Values within physically reasonable ranges
4. **Temporal Continuity**: Regular time intervals with minimal gaps

### 3.3.2 Missing Value Treatment Strategy

**Hierarchical Missing Value Imputation:**

1. **Essential Variables**: Remove records with missing essential meteorological variables
2. **Optional Variables**: Fill missing values with 0 (appropriate for precipitation, wind, UV, solar)
3. **Forward/Backward Fill**: Apply temporal interpolation for remaining missing values
4. **Rolling Mean Imputation**: Use 6-hour rolling mean for persistent gaps
5. **Median Substitution**: Final fallback for any remaining missing values

**Mathematical Formulation:**
```
For missing value at time t:
1. Forward fill: X(t) = X(t-1) if X(t-1) exists
2. Backward fill: X(t) = X(t+1) if X(t+1) exists
3. Rolling mean: X(t) = mean(X(t-n:t+n)) where n=6 hours
4. Median fill: X(t) = median(X) for entire variable
```

### 3.3.3 Outlier Detection and Treatment

**Statistical Outlier Detection:**
```python
# IQR method for outlier detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = 1.5 * IQR
outliers = (df < (Q1 - outlier_threshold)) | (df > (Q3 + outlier_threshold))
```

**Physical Constraint Validation:**
- Temperature: -10°C to 50°C (tropical climate range)
- Humidity: 0% to 100% (physical limits)
- Pressure: 950 hPa to 1050 hPa (sea level range)
- Solar radiation: 0 to 1400 W/m² (theoretical maximum)
- Precipitation: ≥ 0 mm (non-negative values)

### 3.3.4 Temporal Alignment and Sorting

**DateTime Processing:**
```python
# DateTime combination and sorting
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('DateTime')
df = df.reset_index(drop=True)
```

**Temporal Consistency Checks:**
1. **Regular Intervals**: Verify ~30-minute spacing between observations
2. **Chronological Order**: Ensure temporal sequence is maintained
3. **Duplicate Removal**: Eliminate duplicate timestamps
4. **Gap Identification**: Document and address temporal gaps > 2 hours

## 3.4 Feature Engineering Methodology

### 3.4.1 Feature Engineering Framework

The feature engineering process transforms 12 raw meteorological variables into 37 engineered features through systematic application of domain knowledge and temporal modeling techniques.

**Feature Categories:**
1. **Temporal Features** (14 features): Time-based patterns and cycles
2. **Meteorological Derived Features** (3 features): Physics-based relationships
3. **Weather Event Indicators** (3 features): Binary event detection
4. **Rolling Window Features** (10 features): Temporal statistics
5. **Lag Features** (9 features): Historical dependencies

### 3.4.2 Temporal Feature Engineering

**Basic Temporal Components:**
```python
# Temporal feature extraction
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['DayOfYear'] = df['DateTime'].dt.dayofyear
df['Season'] = (df['Month'] % 12 + 3) // 3
```

**Cyclical Encoding Implementation:**
```python
# Cyclical encoding for periodic patterns
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
```

**Mathematical Justification:**
Cyclical encoding preserves the periodic nature of temporal variables, ensuring that:
- Hour 23 is closer to Hour 0 than to Hour 12
- December (Month 12) is closer to January (Month 1) than to June (Month 6)
- Day 365 is closer to Day 1 than to Day 183

### 3.4.3 Meteorological Derived Features

**Temperature-Dew Point Difference:**
```python
# Moisture deficit indicator
df['Temp_Dew_Diff'] = df['Temperature_C'] - df['Dew_Point_C']
```
*Physical Significance*: Indicates atmospheric moisture deficit and potential for condensation/evaporation.

**Wind-Pressure Ratio:**
```python
# Pressure-normalized wind speed
df['Wind_Pressure_Ratio'] = df['Speed_kmh'] / (df['Pressure_hPa'] / 1000)
```
*Physical Significance*: Accounts for pressure effects on wind speed measurements.

**Vapor Pressure Deficit (Simplified):**
```python
# Atmospheric dryness measure
df['Vapor_Pressure_Deficit'] = df['Temp_Dew_Diff'] * 0.1
```
*Physical Significance*: Approximates the difference between saturation and actual vapor pressure.

### 3.4.4 Weather Event Indicators

**Binary Event Detection:**
```python
# Weather event indicators
df['Rain_Event'] = (df['Precip_Rate_mm'] > 0).astype(int)
df['Solar_Active'] = (df['Solar_w/m2'] > 100).astype(int)
df['High_Humidity_Event'] = (df['Humidity_%'] > 80).astype(int)
```

**Threshold Selection Methodology:**
- Rain Event: Any measurable precipitation (>0 mm)
- Solar Active: Significant solar radiation (>100 W/m²)
- High Humidity: Threshold based on tropical climate patterns (>80%)

### 3.4.5 Rolling Window Features

**6-Hour Rolling Statistics:**
```python
# Rolling window implementation
window_size = 12  # 12 records = 6 hours at 30-min intervals
for target in target_variables:
    df[f'{target}_Rolling_Mean_6h'] = df[target].rolling(window=window_size, min_periods=1).mean()
    df[f'{target}_Rolling_Std_6h'] = df[target].rolling(window=window_size, min_periods=1).std()
```

**Mathematical Formulation:**
```
Rolling_Mean(t) = (1/n) × Σ(i=t-n+1 to t) X(i)
Rolling_Std(t) = √[(1/n) × Σ(i=t-n+1 to t) (X(i) - Rolling_Mean(t))²]
```

**Window Size Justification:**
- 6-hour window captures mesoscale weather patterns
- Sufficient for diurnal cycle representation
- Balances temporal memory with computational efficiency

### 3.4.6 Lag Features

**Historical Dependency Modeling:**
```python
# Lag feature implementation
lag_hours = [1, 3, 6]
key_variables = ['Temperature_C', 'Humidity_%', 'Pressure_hPa']

for variable in key_variables:
    for lag_h in lag_hours:
        lag_steps = lag_h * 2  # Convert hours to 30-min steps
        df[f'{variable}_Lag_{lag_h}h'] = df[variable].shift(lag_steps)
```

**Lag Selection Rationale:**
- **1-hour lag**: Captures immediate temporal persistence
- **3-hour lag**: Represents mesoscale weather evolution
- **6-hour lag**: Accounts for diurnal cycle components

## 3.5 Model Development Methodology

### 3.5.1 Model Architecture Design

**Multi-Target Gradient Boosting Framework:**
The research employs a multi-target architecture where separate gradient boosting models are trained for each meteorological variable, allowing for target-specific optimization.

**Model Selection Justification:**
1. **Gradient Boosting Advantages**: Handles non-linear relationships, robust to outliers, provides feature importance
2. **Multi-Target Approach**: Allows target-specific hyperparameter optimization
3. **Interpretability**: Compatible with SHAP analysis for explainable predictions
4. **Computational Efficiency**: Faster than deep learning for tabular data

### 3.5.2 Target-Specific Model Configuration

**Hyperparameter Optimization Strategy:**
Each target variable receives customized hyperparameters based on its characteristics:

```python
model_configs = {
    'Humidity_%': {
        'n_estimators': 150,
        'learning_rate': 0.08,
        'max_depth': 4,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'Temperature_C': {
        'n_estimators': 180,
        'learning_rate': 0.06,
        'max_depth': 5,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'subsample': 0.8,
        'max_features': 'sqrt',
        'random_state': 42
    },
    # ... additional configurations for other targets
}
```

**Configuration Rationale:**
- **Humidity/Temperature**: Higher complexity (more estimators, deeper trees) for continuous variables
- **Precipitation**: Aggressive learning for challenging prediction task
- **Solar Radiation**: Moderate complexity for highly predictable astronomical patterns

### 3.5.3 Multi-Horizon Forecasting Implementation

**Forecast Horizon Definition:**
```python
forecast_horizons = [1, 3, 6, 12, 24]  # Hours ahead
horizon_steps = [h * 2 for h in forecast_horizons]  # Convert to 30-min steps
```

**Target Shifting Methodology:**
```python
# Create future targets for each horizon
for target in target_variables:
    for horizon_h in forecast_horizons:
        steps = horizon_h * 2
        df[f'Future_{target}_{horizon_h}h'] = df[target].shift(-steps)
```

**Separate Model Training:**
Each combination of target variable and forecast horizon receives a dedicated model to optimize prediction accuracy for specific time scales.

### 3.5.4 Feature Scaling and Normalization

**StandardScaler Implementation:**
```python
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mathematical transformation
# X_scaled = (X - μ) / σ
# where μ = mean(X), σ = std(X)
```

**Scaling Justification:**
1. **Gradient Boosting Compatibility**: Improves convergence and stability
2. **Feature Equality**: Prevents dominance by variables with larger scales
3. **Numerical Stability**: Reduces computational precision issues

### 3.5.5 Physical Constraint Implementation

**Post-Prediction Constraints:**
```python
# Physical validity enforcement
def apply_constraints(predictions, target_variable):
    if target_variable == 'Humidity_%':
        return np.clip(predictions, 0, 100)
    elif target_variable in ['Precip_Rate_mm', 'Precip_Accum_mm']:
        return np.maximum(predictions, 0)
    elif target_variable == 'Solar_w/m2':
        return np.maximum(predictions, 0)
    else:
        return predictions
```

**Constraint Rationale:**
- **Humidity**: Physical limits (0-100%)
- **Precipitation**: Non-negative values
- **Solar Radiation**: Non-negative values
- **Temperature**: No constraints (can be negative in some climates)

## 3.6 Validation and Evaluation Methodology

### 3.6.1 Data Splitting Strategy

**Temporal Split Approach:**
```python
# Temporal split for time series data
train_size = int(0.8 * len(data))
X_train = data.iloc[:train_size]
X_test = data.iloc[train_size:]
```

**Split Justification:**
- **80/20 Split**: Provides sufficient training data while maintaining adequate test set
- **Temporal Ordering**: Maintains chronological sequence to prevent data leakage
- **No Random Shuffling**: Preserves temporal dependencies in weather data

### 3.6.2 Cross-Validation Framework

**K-Fold Cross-Validation for Time Series:**
```python
from sklearn.model_selection import KFold

# Time series cross-validation
cv = KFold(n_splits=3, shuffle=False)  # No shuffle for temporal data
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
```

**Cross-Validation Adaptations:**
- **No Shuffling**: Maintains temporal order
- **Reduced Folds**: K=3 to balance validation rigor with computational efficiency
- **Forward Validation**: Each fold uses earlier data for training, later for validation

### 3.6.3 Performance Metrics

**Comprehensive Evaluation Metrics:**

1. **R-squared (Coefficient of Determination):**
   ```
   R² = 1 - (SS_res / SS_tot)
   SS_res = Σ(y_true - y_pred)²
   SS_tot = Σ(y_true - y_mean)²
   ```

2. **Root Mean Squared Error (RMSE):**
   ```
   RMSE = √[(1/n) × Σ(y_true - y_pred)²]
   ```

3. **Mean Absolute Error (MAE):**
   ```
   MAE = (1/n) × Σ|y_true - y_pred|
   ```

**Metric Selection Rationale:**
- **R²**: Measures proportion of variance explained
- **RMSE**: Penalizes large errors, sensitive to outliers
- **MAE**: Robust to outliers, interpretable in original units

### 3.6.4 Baseline Comparison Methodology

**Baseline Models:**
1. **Linear Regression**: Simple linear relationships
2. **Random Forest**: Tree-based ensemble method
3. **Neural Network**: Multi-layer perceptron
4. **ARIMA**: Traditional time series method
5. **Persistence Model**: Naive forecasting baseline

**Comparison Framework:**
```python
# Standardized evaluation across all models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    return metrics
```

### 3.6.5 Robustness Testing

**Seasonal Performance Analysis:**
```python
# Seasonal validation
seasons = {
    'Dry': [4, 5, 6, 7, 8, 9],      # April-September
    'Wet': [10, 11, 12, 1, 2, 3],   # October-March
    'Transition': [3, 4, 9, 10]      # Transition months
}
```

**Data Quality Sensitivity:**
```python
# Missing data sensitivity analysis
missing_percentages = [0, 5, 10, 15, 20]
for missing_pct in missing_percentages:
    # Artificially introduce missing data
    # Evaluate model performance degradation
```

## 3.7 Explainable AI Methodology

### 3.7.1 SHAP Analysis Implementation

**SHAP (SHapley Additive exPlanations) Framework:**
```python
import shap

# SHAP explainer for gradient boosting
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Feature importance calculation
feature_importance = np.abs(shap_values).mean(axis=0)
```

**SHAP Advantages:**
1. **Unified Framework**: Consistent explanations across different models
2. **Local and Global**: Both individual prediction and overall feature importance
3. **Theoretically Grounded**: Based on cooperative game theory
4. **Additive Property**: Explanations sum to prediction difference from baseline

### 3.7.2 Feature Importance Analysis

**Multi-Level Importance Assessment:**
1. **Global Importance**: Average SHAP values across all predictions
2. **Target-Specific Importance**: Feature importance for each meteorological variable
3. **Temporal Importance**: How feature importance varies with forecast horizon
4. **Seasonal Importance**: Feature importance variations across seasons

**Importance Validation:**
```python
# Physical consistency check
def validate_feature_importance(shap_values, feature_names):
    # Check if importance aligns with meteorological knowledge
    # Verify temporal persistence features rank highly
    # Confirm cyclical features important for periodic variables
```

### 3.7.3 Model Interpretability Framework

**Interpretability Dimensions:**
1. **Feature Attribution**: Which features contribute most to predictions
2. **Prediction Confidence**: Uncertainty quantification through prediction intervals
3. **Physical Consistency**: Alignment with meteorological principles
4. **Temporal Patterns**: How explanations change over forecast horizons

## 3.8 Implementation and Computational Considerations

### 3.8.1 Software and Hardware Environment

**Software Stack:**
- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn 1.0+, XGBoost 1.5+
- **Data Processing**: pandas 1.3+, numpy 1.21+
- **Visualization**: matplotlib 3.5+, seaborn 0.11+
- **Explainability**: SHAP 0.41+

**Hardware Requirements:**
- **CPU**: Multi-core processor (8+ cores recommended)
- **Memory**: 16+ GB RAM for large dataset processing
- **Storage**: 50+ GB for data, models, and results

### 3.8.2 Computational Optimization

**Parallel Processing:**
```python
# Multi-core processing configuration
import multiprocessing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)

# Model training with parallel processing
model = GradientBoostingRegressor(n_jobs=N_JOBS)
```

**Memory Management:**
```python
# Efficient data processing
def process_in_chunks(data, chunk_size=10000):
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i+chunk_size]
```

### 3.8.3 Reproducibility Framework

**Random State Management:**
```python
# Ensure reproducible results
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set random states for all models
model_params = {
    'random_state': RANDOM_STATE,
    # ... other parameters
}
```

**Version Control:**
- **Code Versioning**: Git repository with tagged releases
- **Data Versioning**: Checksums for dataset integrity
- **Environment**: Requirements.txt for dependency management

## 3.9 Ethical Considerations and Limitations

### 3.9.1 Data Privacy and Ethics

**Data Usage Ethics:**
- **Public Data**: Utilization of publicly available meteorological data
- **No Personal Information**: Weather data contains no personally identifiable information
- **Attribution**: Proper acknowledgment of data sources
- **Transparency**: Open methodology for reproducibility

### 3.9.2 Methodological Limitations

**Acknowledged Limitations:**
1. **Spatial Scope**: Point-based predictions may not represent broader areas
2. **Temporal Scope**: Model trained on specific time period may not generalize to different climate conditions
3. **Extreme Events**: Limited training data for rare weather events
4. **Physical Processes**: Some sub-grid scale processes not captured in features

### 3.9.3 Uncertainty Quantification

**Uncertainty Sources:**
1. **Model Uncertainty**: Inherent limitations of machine learning models
2. **Data Uncertainty**: Measurement errors and missing data
3. **Feature Uncertainty**: Approximations in derived meteorological features
4. **Temporal Uncertainty**: Increasing uncertainty with forecast horizon

## 3.10 Summary

This comprehensive methodology provides a systematic framework for developing and evaluating the IMAKAS weather prediction algorithm. The approach combines rigorous data preprocessing, physics-informed feature engineering, target-specific model optimization, and comprehensive validation strategies. The integration of explainable AI techniques ensures that the resulting models are not only accurate but also interpretable for operational use.

The methodology's strength lies in its systematic approach to handling the complexities of meteorological prediction while maintaining scientific rigor and practical applicability. The multi-target, multi-horizon framework provides a comprehensive solution for operational weather forecasting needs, while the explainability analysis ensures scientific understanding and operational confidence.

Future applications of this methodology can be extended to different geographical regions, additional meteorological variables, and longer forecast horizons, providing a robust foundation for advancing weather prediction capabilities in tropical and subtropical regions. 