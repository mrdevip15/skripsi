# Methodology Flowcharts for Research Paper

## Simple Academic Flowchart (Best for Paper)

```mermaid
flowchart TD
    A["Raw Weather Data<br/>Makassar Dataset"] --> B["Data Preprocessing<br/>• Handle missing values<br/>• Feature engineering"]
    
    B --> C["Multi-Target Modeling<br/>5 Weather Variables<br/>(RR, ss, Tavg, ddd_car, ff_avg)"]
    
    C --> D["GBM Training<br/>• Separate models per target<br/>• Cross-validation<br/>• Hyperparameter tuning"]
    
    D --> E["Multi-Day Forecasting<br/>0-5 days ahead"]
    
    E --> F["Model Evaluation<br/>• RMSE, MAE, R²<br/>• Feature importance<br/>• Performance analysis"]
    
    F --> G["Results<br/>R² > 0.90 for all targets<br/>Comprehensive visualization"]
```

## Compact Methodology Flowchart (Alternative)

```mermaid
flowchart TD
    A["📊 Raw Weather Data<br/>(Makassar Dataset)"] --> B["🔧 Data Preprocessing"]
    
    B --> B1["Handle Special Values<br/>(8888, 9999)"]
    B1 --> B2["Feature Engineering<br/>(15 features)"]
    B2 --> B3["Temporal Split<br/>(80% train, 20% test)"]
    
    B3 --> C["🎯 Multi-Target Modeling"]
    
    C --> C1["Target 1:<br/>Rainfall (RR)"]
    C --> C2["Target 2:<br/>Sunshine (ss)"]
    C --> C3["Target 3:<br/>Temperature (Tavg)"]
    C --> C4["Target 4:<br/>Wind Direction (ddd_car)"]
    C --> C5["Target 5:<br/>Wind Speed (ff_avg)"]
    
    C1 --> D["🌲 GBM Training<br/>(Separate models per target)"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    
    D --> E["📈 Multi-Day Forecasting<br/>(0-5 days ahead)"]
    
    E --> F["📊 Model Evaluation"]
    F --> F1["Performance Metrics<br/>(RMSE, MAE, R²)"]
    F --> F2["Feature Importance<br/>(Top 5 per target)"]
    F --> F3["Cross-Validation<br/>(Time series split)"]
    
    F1 --> G["📋 Results & Visualization"]
    F2 --> G
    F3 --> G
    
    G --> G1["Individual Predictions<br/>(Figures 6a-6e)"]
    G --> G2["Multi-Day Forecasts<br/>(Figures 7a-7e)"]
    G --> G3["Performance Analysis<br/>(Figures 8-9)"]
    
    G1 --> H["✅ Research Outputs<br/>R² > 0.90 for all targets"]
    G2 --> H
    G3 --> H
```

## Detailed Technical Flowchart (For Appendix/Supplementary)

```mermaid
flowchart TD
    A["📊 Input Data<br/>Makassar Weather Dataset<br/>(2019-2024)"] --> B["🔧 Data Preprocessing"]
    
    B --> B1["Missing Value Handling<br/>• Forward/Backward Fill<br/>• 5-day Rolling Mean<br/>• Median Imputation"]
    B1 --> B2["Feature Engineering<br/>• Temporal: Month_sin/cos<br/>• Meteorological: Dew_Point<br/>• Lag Features: RR_Lag_1,2<br/>• Rolling Stats: 3d averages"]
    B2 --> B3["Data Splitting<br/>Temporal Split (80/20)<br/>Preserve time order"]
    
    B3 --> C["🎯 Multi-Target Setup<br/>5 Weather Variables"]
    
    C --> D["🌲 GBM Model Training<br/>Gradient Boosting Machine"]
    
    D --> D1["Target-Specific Hyperparameters<br/>• n_estimators: 150-200<br/>• learning_rate: 0.05-0.08<br/>• max_depth: 4-5<br/>• Cross-validation: 3-5 folds"]
    
    D1 --> E["📈 Multi-Horizon Forecasting"]
    
    E --> E1["Day 0: Same-day prediction"]
    E --> E2["Day 1-5: Multi-day ahead"]
    
    E1 --> F["📊 Model Evaluation"]
    E2 --> F
    
    F --> F1["Performance Metrics<br/>• RMSE, MAE, R²<br/>• Cross-validation scores<br/>• Constraint application"]
    
    F1 --> G["📈 Feature Analysis<br/>• Importance ranking<br/>• Correlation analysis<br/>• Seasonal patterns"]
    
    G --> H["📋 Visualization & Results<br/>• 9 main figures<br/>• 12 supplementary plots<br/>• Performance heatmaps"]
    
    H --> I["✅ Final Output<br/>R² Scores:<br/>• Rainfall: 0.901<br/>• Sunshine: 0.982<br/>• Temperature: 0.923<br/>• Wind Direction: 0.978<br/>• Wind Speed: 0.946"]
```

## Usage Instructions for Research Paper

### For Main Paper (Methodology Section):
- **Use**: Simple Academic Flowchart (recommended)
- **Caption**: "Methodology flowchart for multi-target weather prediction using GBM. The system processes raw weather data through preprocessing and feature engineering, trains separate GBM models for five weather variables, and evaluates performance across 0-5 day forecast horizons."
- **Placement**: Early in Methodology section to provide clear overview

### For Supplementary Material:
- **Use**: Detailed Technical Flowchart  
- **Caption**: "Detailed technical flowchart showing specific algorithms, hyperparameters, and evaluation metrics used in the multi-target GBM weather prediction system."
- **Placement**: Appendix or supplementary methods section

## Key Process Steps Highlighted:

1. **Data Preprocessing**: Comprehensive handling of meteorological data quirks
2. **Feature Engineering**: 15 carefully selected features including temporal and lag components
3. **Multi-Target Approach**: Separate GBM models for each weather variable
4. **Multi-Day Forecasting**: Systematic evaluation across 0-5 day horizons
5. **Robust Evaluation**: Multiple metrics with cross-validation
6. **Comprehensive Visualization**: 21 figures for complete analysis

## Technical Specifications:

- **Algorithm**: Gradient Boosting Machine (scikit-learn)
- **Targets**: 5 weather variables (RR, ss, Tavg, ddd_car, ff_avg)
- **Features**: 15 engineered features from meteorological data
- **Evaluation**: Time series cross-validation with temporal split
- **Performance**: R² > 0.90 for all targets
- **Forecast**: Up to 5 days ahead with graceful degradation

## Figure Reference:
- Save as **Figure 10** in your paper: "Methodology Flowchart"
- High-resolution versions can be exported from Mermaid or recreated in your preferred diagramming tool
- Consider using consistent colors with your other figures

---

*These flowcharts provide a clear visual summary of your entire methodology, perfect for helping readers understand your comprehensive approach to multi-target weather prediction.* 