# IMAKAS Weather Prediction Algorithm - Technical Summary

## Overview
The IMAKAS Weather Prediction Algorithm is a multi-target, multi-horizon gradient boosting machine learning system designed to predict five key weather variables using high-frequency meteorological data. The system employs advanced feature engineering, temporal modeling, and explainable AI techniques.

## 1. Target Variables

The algorithm predicts five meteorological variables:

| Variable | Indonesian Name | Unit | Range | Description |
|----------|----------------|------|-------|-------------|
| `Humidity_%` | Kelembaban | % | 0-100 | Relative humidity percentage |
| `Temperature_C` | Suhu | °C | -∞ to +∞ | Air temperature in Celsius |
| `Precip_Rate_mm` | Laju Curah Hujan | mm | ≥0 | Precipitation rate |
| `Precip_Accum_mm` | Akumulasi Curah Hujan | mm | ≥0 | Accumulated precipitation |
| `Solar_w/m2` | Radiasi Matahari | W/m² | ≥0 | Solar radiation intensity |

## 2. Input Data Structure

### 2.1 Raw Data Format
- **Source**: IMAKAS7.csv (204,073 records)
- **Temporal Resolution**: ~30-minute intervals
- **Time Span**: March 2023 to May 2025
- **Variables**: Date, Time, Temperature_C, Dew_Point_C, Humidity_%, Wind, Speed_kmh, Gust_kmh, Pressure_hPa, Precip_Rate_mm, Precip_Accum_mm, UV, Solar_w/m2

### 2.2 Data Preprocessing

#### DateTime Processing
```python
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('DateTime')
```

#### Missing Value Handling
1. **Essential variables**: Remove rows with missing Temperature_C, Humidity_%, Pressure_hPa, Dew_Point_C
2. **Optional variables**: Fill with 0 for Speed_kmh, Gust_kmh, Precip_Rate_mm, Precip_Accum_mm, UV, Solar_w/m2
3. **Forward/backward fill**: For remaining missing values

## 3. Feature Engineering

### 3.1 Temporal Features

#### Basic Time Components
```python
df['Hour'] = df['DateTime'].dt.hour          # 0-23
df['Day'] = df['DateTime'].dt.day            # 1-31
df['Month'] = df['DateTime'].dt.month        # 1-12
df['DayOfWeek'] = df['DateTime'].dt.dayofweek # 0-6
df['DayOfYear'] = df['DateTime'].dt.dayofyear # 1-365
df['Season'] = (df['Month'] % 12 + 3) // 3   # 1-4
```

#### Cyclical Encoding
To capture periodic patterns, temporal features are encoded using sine and cosine transformations:

**Hour Encoding:**
```
Hour_sin = sin(2π × Hour / 24)
Hour_cos = cos(2π × Hour / 24)
```

**Day Encoding:**
```
Day_sin = sin(2π × Day / 31)
Day_cos = cos(2π × Day / 31)
```

**Month Encoding:**
```
Month_sin = sin(2π × Month / 12)
Month_cos = cos(2π × Month / 12)
```

### 3.2 Meteorological Derived Features

#### Temperature-Dew Point Difference
```
Temp_Dew_Diff = Temperature_C - Dew_Point_C
```
*Physical significance*: Indicates atmospheric moisture deficit and potential for condensation.

#### Wind-Pressure Ratio
```
Wind_Pressure_Ratio = Speed_kmh / (Pressure_hPa / 1000)
```
*Physical significance*: Normalized wind speed accounting for atmospheric pressure effects.

#### Vapor Pressure Deficit (Simplified)
```
Vapor_Pressure_Deficit = Temp_Dew_Diff × 0.1
```
*Physical significance*: Approximates the difference between saturation vapor pressure and actual vapor pressure.

### 3.3 Weather Event Indicators

Binary indicators for significant weather conditions:

```python
Rain_Event = (Precip_Rate_mm > 0).astype(int)
Solar_Active = (Solar_w/m2 > 100).astype(int)
High_Humidity_Event = (Humidity_% > 80).astype(int)
```

### 3.4 Rolling Window Features

6-hour rolling statistics (window = 12 records assuming 30-min intervals):

**For each target variable:**
```
Variable_Rolling_Mean_6h = rolling_mean(Variable, window=12)
Variable_Rolling_Std_6h = rolling_std(Variable, window=12)
```

**Mathematical formulation:**
```
Rolling_Mean(t) = (1/n) × Σ(i=t-n+1 to t) X(i)
Rolling_Std(t) = √[(1/n) × Σ(i=t-n+1 to t) (X(i) - Rolling_Mean(t))²]
```

### 3.5 Lag Features

Historical values at different time steps:

**Lag Steps:**
- 1 hour lag: `Variable_Lag_1h = Variable.shift(2)`  # 2 records = 1 hour
- 3 hour lag: `Variable_Lag_3h = Variable.shift(6)`  # 6 records = 3 hours  
- 6 hour lag: `Variable_Lag_6h = Variable.shift(12)` # 12 records = 6 hours

**Applied to:** Temperature_C, Humidity_%, Pressure_hPa

## 4. Model Architecture

### 4.1 Gradient Boosting Regressor Configuration

**Target-Specific Hyperparameters:**

```python
model_configs = {
    'Humidity_%': GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=4,
        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
        max_features='sqrt', random_state=42
    ),
    'Temperature_C': GradientBoostingRegressor(
        n_estimators=180, learning_rate=0.06, max_depth=5,
        min_samples_split=8, min_samples_leaf=4, subsample=0.8,
        max_features='sqrt', random_state=42
    ),
    'Precip_Rate_mm': GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        max_features='sqrt', random_state=42
    ),
    'Precip_Accum_mm': GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        max_features='sqrt', random_state=42
    ),
    'Solar_w/m2': GradientBoostingRegressor(
        n_estimators=160, learning_rate=0.07, max_depth=5,
        min_samples_split=6, min_samples_leaf=4, subsample=0.8,
        max_features='sqrt', random_state=42
    )
}
```

### 4.2 Gradient Boosting Mathematical Framework

**Objective Function:**
```
L(y, F(x)) = Σ(i=1 to n) l(yi, F(xi)) + Ω(F)
```

Where:
- `l(yi, F(xi))` is the loss function (MSE for regression)
- `Ω(F)` is the regularization term
- `F(x)` is the ensemble prediction

**Boosting Algorithm:**
```
F0(x) = argmin(γ) Σ(i=1 to n) l(yi, γ)

For m = 1 to M:
    1. Compute negative gradients: rim = -[∂l(yi, F(xi))/∂F(xi)]|F=Fm-1
    2. Fit regression tree to rim: {Rjm}(j=1 to Jm)
    3. Compute terminal node values: γjm = argmin(γ) Σ(xi∈Rjm) l(yi, Fm-1(xi) + γ)
    4. Update: Fm(x) = Fm-1(x) + ν × Σ(j=1 to Jm) γjm × I(x ∈ Rjm)

Final model: F(x) = FM(x)
```

### 4.3 Feature Scaling

**StandardScaler Transformation:**
```
X_scaled = (X - μ) / σ

Where:
μ = (1/n) × Σ(i=1 to n) xi  (mean)
σ = √[(1/n) × Σ(i=1 to n) (xi - μ)²]  (standard deviation)
```

### 4.4 Target Variable Constraints

Post-prediction constraints to ensure physical validity:

```python
if target == 'Humidity_%':
    predictions = np.clip(predictions, 0, 100)
elif target in ['Precip_Rate_mm', 'Precip_Accum_mm']:
    predictions = np.maximum(predictions, 0)
elif target == 'Solar_w/m2':
    predictions = np.maximum(predictions, 0)
```

## 5. Multi-Horizon Forecasting

### 5.1 Forecast Horizons
- 1 hour ahead (2 time steps)
- 3 hours ahead (6 time steps)
- 6 hours ahead (12 time steps)
- 12 hours ahead (24 time steps)
- 24 hours ahead (48 time steps)

### 5.2 Target Shifting
```python
# For h-hour ahead prediction:
steps = hours × 2  # Assuming 30-min intervals
df[f'Future_{target}_{hours}h'] = df[target].shift(-steps)
```

## 6. Model Training and Validation

### 6.1 Temporal Split Strategy
```python
train_size = int(0.8 × len(data))
X_train = data.iloc[:train_size]
X_test = data.iloc[train_size:]
```

### 6.2 Cross-Validation
**K-Fold Cross-Validation (K=3):**
```python
cv = KFold(n_splits=3, shuffle=False)  # No shuffle for time series
```

**Cross-validation score:**
```
CV_Score = (1/K) × Σ(k=1 to K) Score(k)
CV_Std = √[(1/K) × Σ(k=1 to K) (Score(k) - CV_Score)²]
```

## 7. Evaluation Metrics

### 7.1 Regression Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(i=1 to n) (yi - ŷi)²
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE = √[(1/n) × Σ(i=1 to n) (yi - ŷi)²]
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ(i=1 to n) |yi - ŷi|
```

**R-squared (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ(i=1 to n) (yi - ŷi)²  (Residual sum of squares)
SS_tot = Σ(i=1 to n) (yi - ȳ)²   (Total sum of squares)
ȳ = (1/n) × Σ(i=1 to n) yi      (Mean of observed values)
```

### 7.2 Performance Ranges
- **R² > 0.9**: Excellent prediction
- **0.7 < R² < 0.9**: Good prediction  
- **0.5 < R² < 0.7**: Moderate prediction
- **R² < 0.5**: Poor prediction

## 8. SHAP (SHapley Additive exPlanations) Analysis

### 8.1 SHAP Value Calculation

**Shapley Value Formula:**
```
φi = Σ(S⊆N\{i}) [|S|!(|N|-|S|-1)!/|N|!] × [f(S∪{i}) - f(S)]
```

Where:
- `φi` is the SHAP value for feature i
- `N` is the set of all features
- `S` is a subset of features not including i
- `f(S)` is the model prediction using feature subset S

### 8.2 TreeExplainer for Gradient Boosting

For tree-based models, SHAP uses an efficient algorithm:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
```

**Additivity Property:**
```
f(x) = E[f(X)] + Σ(i=1 to p) φi(x)
```

Where `E[f(X)]` is the expected model output and `φi(x)` are the SHAP values.

## 9. Algorithm Workflow

### 9.1 Training Phase

```
1. Data Loading and Preprocessing
   ├── Load IMAKAS7.csv
   ├── Convert DateTime
   ├── Handle missing values
   └── Sort by timestamp

2. Feature Engineering
   ├── Extract temporal features
   ├── Apply cyclical encoding
   ├── Compute derived meteorological features
   ├── Create weather event indicators
   ├── Generate rolling window features
   └── Create lag features

3. Model Training (for each target)
   ├── Split data temporally (80% train, 20% test)
   ├── Scale features using StandardScaler
   ├── Configure target-specific GBM parameters
   ├── Train with K-fold cross-validation
   ├── Apply post-prediction constraints
   └── Evaluate performance metrics

4. Multi-Horizon Training
   ├── Create shifted targets for each horizon
   ├── Train separate models for each horizon
   └── Store models and scalers

5. Analysis and Visualization
   ├── Generate prediction plots with metrics
   ├── Create accuracy vs horizon analysis
   ├── Perform SHAP analysis
   └── Generate comprehensive reports
```

### 9.2 Prediction Phase

```
1. Input Processing
   ├── Preprocess new data
   ├── Engineer features
   └── Scale using trained scaler

2. Multi-Target Prediction
   ├── Load appropriate model for each target
   ├── Generate predictions
   └── Apply physical constraints

3. Multi-Horizon Forecasting
   ├── Use horizon-specific models
   ├── Generate forecasts for 1h, 3h, 6h, 12h, 24h
   └── Combine results
```

## 10. Key Innovation and Contributions

### 10.1 Technical Innovations
1. **Multi-target, multi-horizon architecture** with target-specific hyperparameters
2. **Comprehensive temporal feature engineering** with cyclical encoding
3. **Physics-informed derived features** (VPD, wind-pressure ratio)
4. **Robust handling of high-frequency meteorological data**
5. **Integrated explainable AI** with SHAP analysis

### 10.2 Meteorological Considerations
1. **Physical constraints** ensure realistic predictions
2. **Temporal dependencies** captured through lag and rolling features
3. **Seasonal patterns** modeled with cyclical encoding
4. **Weather event detection** through binary indicators

### 10.3 Performance Characteristics
- **Temperature**: R² > 0.98 (1-3h), R² > 0.68 (24h)
- **Humidity**: R² > 0.97 (1h), R² > 0.61 (24h)  
- **Solar Radiation**: R² > 0.99 (excellent diurnal pattern capture)
- **Precipitation**: R² 0.25-0.40 (challenging but typical for meteorology)

## 11. Computational Complexity

### 11.1 Training Complexity
- **Time Complexity**: O(n × m × d × T) where:
  - n = number of samples (~200k)
  - m = number of features (37)
  - d = tree depth (4-6)
  - T = number of trees (150-200)

### 11.2 Memory Requirements
- **Feature Matrix**: ~200k × 37 × 8 bytes ≈ 59 MB
- **Model Storage**: ~5 models × 5 horizons × ~10 MB ≈ 250 MB
- **SHAP Analysis**: Additional ~100 MB for explainability

## 12. Conclusion

The IMAKAS Weather Prediction Algorithm represents a comprehensive machine learning solution for multi-variate weather forecasting. By combining advanced feature engineering, target-specific modeling, and explainable AI techniques, it achieves excellent performance for temperature, humidity, and solar radiation predictions while providing valuable insights into model behavior through SHAP analysis.

The algorithm's strength lies in its ability to capture complex temporal dependencies, meteorological relationships, and seasonal patterns while maintaining computational efficiency and providing interpretable results suitable for both operational forecasting and research applications. 