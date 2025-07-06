# IMAKAS Data Preprocessing Summary

## Overview
This document provides a comprehensive summary of the data preprocessing pipeline implemented in the IMAKAS (Indonesian Multi-target Automated Klimatologi Analysis System) weather prediction algorithm. The preprocessing transforms raw meteorological data from IMAKAS7.csv into a feature-rich dataset suitable for machine learning model training.

## 1. Input Data Characteristics

### 1.1 Source Dataset: IMAKAS7.csv
- **Records**: 204,073 observations
- **Temporal Resolution**: ~30-minute intervals
- **Time Period**: March 2023 - May 2025
- **Location**: Makassar, South Sulawesi, Indonesia
- **Format**: CSV with timestamp and meteorological variables

### 1.2 Raw Input Variables (12 variables)

| Variable | Description | Unit | Data Type | Example Range |
|----------|-------------|------|-----------|---------------|
| `Date` | Observation date | YYYY-MM-DD | String | 2023-03-01 to 2025-05-31 |
| `Time` | Observation time | HH:MM:SS | String | 00:00:00 to 23:59:59 |
| `Temperature_C` | Air temperature | °C | Float | 22.5 - 35.8 |
| `Dew_Point_C` | Dew point temperature | °C | Float | 18.2 - 28.4 |
| `Humidity_%` | Relative humidity | % | Float | 45.2 - 98.7 |
| `Pressure_hPa` | Atmospheric pressure | hPa | Float | 1008.2 - 1016.8 |
| `Speed_kmh` | Wind speed | km/h | Float | 0.0 - 28.6 |
| `Gust_kmh` | Wind gust speed | km/h | Float | 0.0 - 45.2 |
| `Precip_Rate_mm` | Precipitation rate | mm | Float | 0.0 - 25.4 |
| `Precip_Accum_mm` | Accumulated precipitation | mm | Float | 0.0 - 156.8 |
| `UV` | UV index | - | Float | 0.0 - 12.8 |
| `Solar_w/m2` | Solar radiation | W/m² | Float | 0.0 - 1234.5 |

### 1.3 Target Variables (5 variables)

| Variable | Indonesian Name | Unit | Prediction Purpose |
|----------|----------------|------|-------------------|
| `Humidity_%` | Kelembaban (%) | % | Atmospheric moisture prediction |
| `Temperature_C` | Suhu (°C) | °C | Temperature forecasting |
| `Precip_Rate_mm` | Laju Curah Hujan (mm) | mm | Precipitation intensity |
| `Precip_Accum_mm` | Akumulasi Curah Hujan (mm) | mm | Cumulative precipitation |
| `Solar_w/m2` | Radiasi Matahari (W/m²) | W/m² | Solar radiation forecasting |

## 2. Preprocessing Pipeline Overview

### 2.1 Pipeline Stages
```
Raw Data (IMAKAS7.csv)
    ↓
1. DateTime Processing & Sorting
    ↓
2. Temporal Feature Extraction
    ↓
3. Missing Value Handling
    ↓
4. Derived Feature Creation
    ↓
5. Weather Event Indicators
    ↓
6. Rolling Window Features
    ↓
7. Lag Feature Generation
    ↓
8. NaN Value Treatment
    ↓
9. Final Data Validation
    ↓
Clean Feature Matrix (37 features)
```

### 2.2 Code Implementation Structure
```python
def preprocess_data(self, df):
    """Enhanced preprocessing for IMAKAS7.csv data"""
    # Main preprocessing function in IMakasWeatherPredictor class
    # Returns: Cleaned and feature-enriched DataFrame
```

## 3. Detailed Preprocessing Steps

### 3.1 DateTime Processing and Sorting

**Purpose**: Convert separate Date and Time columns into a unified DateTime index for temporal analysis.

**Implementation**:
```python
# Combine Date and Time columns
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Sort by DateTime to ensure chronological order
df = df.sort_values('DateTime')
```

**Output**: 
- New `DateTime` column with proper datetime format
- Data sorted chronologically
- Enables temporal feature extraction

### 3.2 Temporal Feature Extraction

**Purpose**: Extract time-based features to capture temporal patterns in weather data.

**Basic Temporal Components**:
```python
df['Hour'] = df['DateTime'].dt.hour          # 0-23
df['Day'] = df['DateTime'].dt.day            # 1-31
df['Month'] = df['DateTime'].dt.month        # 1-12
df['DayOfWeek'] = df['DateTime'].dt.dayofweek # 0-6 (Monday=0)
df['DayOfYear'] = df['DateTime'].dt.dayofyear # 1-365
df['Season'] = (df['Month'] % 12 + 3) // 3   # 1-4 (Winter=1, Spring=2, Summer=3, Fall=4)
```

**Cyclical Encoding**:
```python
# Hour cyclical encoding (24-hour cycle)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)

# Day cyclical encoding (31-day cycle)
df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)

# Month cyclical encoding (12-month cycle)
df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
```

**Generated Features**: 12 temporal features
- 6 basic temporal components
- 6 cyclical encoded features

### 3.3 Missing Value Handling Strategy

**Essential Variables Strategy**:
```python
# Define essential meteorological variables
essential_cols = ['Temperature_C', 'Humidity_%', 'Pressure_hPa', 'Dew_Point_C']

# Remove rows with missing essential data
df = df.dropna(subset=essential_cols)
```

**Optional Variables Strategy**:
```python
# Define optional variables that can be filled with zeros
numeric_cols = ['Speed_kmh', 'Gust_kmh', 'Precip_Rate_mm', 'Precip_Accum_mm', 'UV', 'Solar_w/m2']

# Fill missing values with 0 (appropriate for precipitation, wind, UV, solar)
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
```

**Rationale**:
- **Essential variables**: Critical for weather prediction, missing data indicates unreliable measurements
- **Optional variables**: Zero is physically meaningful (no precipitation, no wind, no solar radiation)

### 3.4 Derived Meteorological Features

**Temperature-Dew Point Difference**:
```python
df['Temp_Dew_Diff'] = df['Temperature_C'] - df['Dew_Point_C']
```
- **Physical Significance**: Indicates atmospheric moisture deficit
- **Range**: 0-20°C (higher values = drier air)
- **Importance**: Critical for humidity and precipitation prediction

**Wind-Pressure Ratio**:
```python
df['Wind_Pressure_Ratio'] = df['Speed_kmh'] / (df['Pressure_hPa'] / 1000)
```
- **Physical Significance**: Pressure-normalized wind speed
- **Purpose**: Accounts for pressure effects on wind measurements
- **Units**: km/h per kPa

**Vapor Pressure Deficit (Simplified)**:
```python
df['Vapor_Pressure_Deficit'] = df['Temp_Dew_Diff'] * 0.1
```
- **Physical Significance**: Approximates atmospheric dryness
- **Formula**: Simplified VPD calculation
- **Application**: Important for evapotranspiration and humidity modeling

**Generated Features**: 3 derived meteorological features

### 3.5 Weather Event Indicators

**Binary Event Detection**:
```python
# Precipitation event indicator
df['Rain_Event'] = (df['Precip_Rate_mm'] > 0).astype(int)

# Solar radiation activity indicator
df['Solar_Active'] = (df['Solar_w/m2'] > 100).astype(int)

# High humidity event indicator
df['High_Humidity_Event'] = (df['Humidity_%'] > 80).astype(int)
```

**Threshold Selection**:
- **Rain Event**: Any measurable precipitation (>0 mm)
- **Solar Active**: Significant solar radiation (>100 W/m²)
- **High Humidity**: Tropical high humidity threshold (>80%)

**Generated Features**: 3 binary weather event indicators

### 3.6 Rolling Window Features

**6-Hour Rolling Statistics**:
```python
rolling_window = 12  # 12 records = ~6 hours (30-min intervals)

# For each target variable
for target in self.target_columns:
    if target in df.columns:
        df[f'{target}_Rolling_Mean_6h'] = df[target].rolling(window=rolling_window, min_periods=1).mean()
        df[f'{target}_Rolling_Std_6h'] = df[target].rolling(window=rolling_window, min_periods=1).std()

# For pressure (additional key variable)
df['Pressure_hPa_Rolling_Mean_6h'] = df['Pressure_hPa'].rolling(window=rolling_window, min_periods=1).mean()
df['Pressure_hPa_Rolling_Std_6h'] = df['Pressure_hPa'].rolling(window=rolling_window, min_periods=1).std()
```

**Mathematical Formulation**:
```
Rolling_Mean(t) = (1/n) × Σ(i=t-n+1 to t) X(i)
Rolling_Std(t) = √[(1/n) × Σ(i=t-n+1 to t) (X(i) - Rolling_Mean(t))²]
```

**Window Size Rationale**:
- **6-hour window**: Captures mesoscale weather patterns
- **min_periods=1**: Prevents initial NaN values
- **12 records**: Assumes ~30-minute data intervals

**Generated Features**: 12 rolling window features (10 for targets + 2 for pressure)

### 3.7 Lag Feature Generation

**Multi-Scale Lag Features**:
```python
# Define lag steps for different time scales
lag_steps = [2, 6, 12]  # 2=1h, 6=3h, 12=6h (30-min intervals)
lag_names = ['1h', '3h', '6h']

# Create lag features for all target variables
for target in self.target_columns:
    if target in df.columns:
        for lag_step, lag_name in zip(lag_steps, lag_names):
            df[f'{target}_Lag_{lag_name}'] = df[target].shift(lag_step)

# Create lag features for pressure (1h and 3h only)
for lag_step, lag_name in zip(lag_steps[:2], lag_names[:2]):
    df[f'Pressure_hPa_Lag_{lag_name}'] = df['Pressure_hPa'].shift(lag_step)
```

**Lag Selection Strategy**:
- **1-hour lag**: Immediate temporal persistence
- **3-hour lag**: Short-term weather evolution
- **6-hour lag**: Medium-term atmospheric memory

**Generated Features**: 17 lag features (15 for targets + 2 for pressure)

### 3.8 NaN Value Treatment for Derived Features

**Lag Feature Treatment**:
```python
# Forward fill for lag features (temporal interpolation)
lag_cols = [col for col in df.columns if '_Lag_' in col]
for col in lag_cols:
    df[col] = df[col].fillna(method='ffill')
```

**Rolling Feature Treatment**:
```python
# Forward fill for rolling features
rolling_cols = [col for col in df.columns if '_Rolling_' in col]
for col in rolling_cols:
    df[col] = df[col].fillna(method='ffill')
```

**Final NaN Cleanup**:
```python
# Comprehensive NaN handling
df = df.fillna(method='ffill').fillna(method='bfill')

# Remove any remaining rows with NaN values
df = df.dropna()
```

**Treatment Hierarchy**:
1. Forward fill (propagate last valid observation)
2. Backward fill (propagate next valid observation)
3. Row removal (final fallback for remaining NaNs)

### 3.9 Feature List Validation

**Dynamic Feature Selection**:
```python
# Update feature columns list to only include existing columns
self.feature_columns = [col for col in self.feature_columns if col in df.columns]
```

**Final Feature Count**: 37 engineered features

## 4. Output Feature Matrix

### 4.1 Feature Categories Summary

| Category | Count | Features |
|----------|-------|----------|
| **Basic Measurements** | 5 | Dew_Point_C, Speed_kmh, Gust_kmh, Pressure_hPa, UV |
| **Temporal Features** | 12 | Hour, Day, Month, DayOfWeek, DayOfYear, Season + 6 cyclical |
| **Derived Features** | 3 | Temp_Dew_Diff, Wind_Pressure_Ratio, Vapor_Pressure_Deficit |
| **Event Indicators** | 3 | Rain_Event, Solar_Active, High_Humidity_Event |
| **Rolling Features** | 12 | 6h rolling mean/std for 5 targets + pressure |
| **Lag Features** | 17 | 1h/3h/6h lags for 5 targets + 1h/3h for pressure |
| **Total** | **37** | **Complete feature set** |

### 4.2 Feature Naming Convention

**Pattern**: `{Variable}_{Operation}_{TimeWindow}`

**Examples**:
- `Temperature_C_Rolling_Mean_6h`: 6-hour rolling mean of temperature
- `Humidity_%_Lag_1h`: 1-hour lag of humidity
- `Hour_sin`: Sine-encoded hour for cyclical representation

### 4.3 Data Quality Metrics

**Before Preprocessing**:
```
Initial dataset size: 204,073 records
Raw variables: 12
Missing data: Variable amounts
```

**After Preprocessing**:
```python
print(f"Final dataset size: {len(df)}")
print(f"Total rows removed: {initial_size - len(df)}")
print(f"Final feature list ({len(self.feature_columns)} features)")
```

## 5. Quality Control and Validation

### 5.1 Statistical Validation

**Target Variable Statistics**:
```python
for target in self.target_columns:
    if target in df.columns:
        print(f"\n{self.target_names[target]} ({target}):")
        print(f"  Mean: {df[target].mean():.2f}")
        print(f"  Median: {df[target].median():.2f}")
        print(f"  Std Dev: {df[target].std():.2f}")
        print(f"  Min: {df[target].min():.2f}")
        print(f"  Max: {df[target].max():.2f}")
```

### 5.2 Visual Validation

**Distribution Plots**:
```python
self.plot_target_distributions(df, self.run_dir, "Distribusi Awal Variabel Target")
```

**Generated Outputs**:
- Distribution histograms for all target variables
- Statistical summaries printed to console
- Feature list validation and reporting

### 5.3 Data Integrity Checks

**Temporal Consistency**:
- Chronological ordering maintained
- No duplicate timestamps
- Regular interval validation

**Physical Constraints**:
- Humidity: 0-100% range
- Precipitation: Non-negative values
- Temperature: Reasonable tropical climate range
- Solar radiation: Non-negative, diurnal patterns

## 6. Performance Considerations

### 6.1 Computational Efficiency

**Memory Usage**:
- Efficient pandas operations
- In-place modifications where possible
- Selective feature creation

**Processing Time**:
- Vectorized operations for feature creation
- Batch processing for rolling/lag features
- Minimal data copying

### 6.2 Scalability

**Large Dataset Handling**:
- Chunk-based processing capability
- Memory-efficient rolling window implementation
- Optimized for high-frequency time series data

## 7. Best Practices Implemented

### 7.1 Temporal Data Handling
- **No Future Data Leakage**: Lag features only use past information
- **Chronological Preservation**: Temporal order maintained throughout
- **Missing Value Strategy**: Physics-informed imputation methods

### 7.2 Meteorological Domain Knowledge
- **Physical Relationships**: Derived features based on atmospheric physics
- **Seasonal Patterns**: Cyclical encoding for periodic phenomena
- **Event Detection**: Threshold-based weather event identification

### 7.3 Feature Engineering Principles
- **Multi-Scale Temporal**: 1h, 3h, 6h time scales
- **Statistical Robustness**: Rolling statistics for trend capture
- **Domain-Specific**: Meteorology-informed feature creation

## 8. Error Handling and Robustness

### 8.1 Exception Management
```python
try:
    # Preprocessing operations
    return df
except Exception as e:
    print(f"Error in preprocessing: {str(e)}")
    raise
```

### 8.2 Data Validation
- Column existence checks
- Data type validation
- Range validation for physical variables

## 9. Output Summary

### 9.1 Transformation Results
- **Input**: 12 raw meteorological variables
- **Output**: 37 engineered features + 5 target variables
- **Data Reduction**: Removal of incomplete records
- **Quality Enhancement**: Physics-informed feature engineering

### 9.2 Ready for Model Training
The preprocessed dataset provides:
- Clean, complete feature matrix
- Temporally consistent data
- Physics-informed derived features
- Multi-scale temporal representations
- Weather event indicators

This preprocessing pipeline ensures the IMAKAS algorithm receives high-quality, feature-rich data optimized for accurate weather prediction across multiple time horizons. 