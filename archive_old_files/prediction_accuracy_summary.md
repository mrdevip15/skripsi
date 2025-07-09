# Weather Prediction Model Accuracy Summary

## Overview
This document provides a comprehensive summary of prediction accuracies for all models and target variables used in the weather prediction project.

## Model Performance Summary

### 1. Multi-Target GBM Models (Latest Results - June 2025)
**Best performing models for specific weather parameters:**

| Target Variable | Full Name | Units | RMSE | MAE | R² | MSE |
|---|---|---|---|---|---|---|
| RR | Rainfall | mm | 6.808 | 1.881 | **0.9007** | 46.352 |
| Tavg | Average Temperature | °C | 0.316 | 0.205 | **0.9232** | 0.100 |
| ff_avg | Wind Speed | m/s | 0.152 | 0.049 | **0.9461** | 0.023 |
| ss | Sunshine Duration | hours | 0.439 | 0.325 | **0.9824** | 0.193 |
| ddd_car | Wind Direction | degrees | 13.292 | 4.894 | **0.9778** | 176.666 |

### 2. Advanced GBM Multi-Day Forecasts (May 2025)
**Performance by forecast horizon (Best run: 2025-05-19 22:22:19):**

| Forecast Day | RMSE | MAE | R² | MAPE | Explained Variance |
|---|---|---|---|---|---|
| Day 1 (Tomorrow) | 11.153 | 7.329 | **0.3953** | 2.18×10¹² | 0.4061 |
| Day 2 | 13.098 | 8.924 | **0.1617** | 3.06×10¹² | 0.1766 |
| Day 3 | 13.163 | 9.024 | **0.1534** | 3.12×10¹² | 0.1745 |
| **Average** | **12.471** | **8.426** | **0.2368** | **2.79×10¹²** | **0.2524** |

### 3. Rainfall Specialist Model (May 2025)
**Specialized rainfall prediction with different feature sets:**

| Forecast Day | Feature Set | Features Count | RMSE | R² |
|---|---|---|---|---|
| Day 1 | All features | 77 | 12.146 | **0.2717** |
| Day 2 | Rainfall patterns | 40 | 14.324 | -0.0129 |
| Day 3 | Rainfall patterns | 40 | 14.029 | 0.0282 |
| **Average** | - | - | **13.500** | **0.0957** |

### 4. Traditional ML Models Comparison (Archive Results)
**Single-day rainfall prediction comparison:**

| Model | Type | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|---|
| **GBM** | Tree-based | **0.056** | **0.002** | **0.9976** | **0.050** |
| Random Forest | Tree-based | 0.187 | 0.019 | 0.9731 | 5.740 |
| SVM | Kernel-based | 0.314 | 0.044 | 0.9239 | 19.379 |
| LSTM | Deep Learning | 1.132 | 0.551 | 0.0126 | 117.302 |
| CNN | Deep Learning | 1.138 | 0.551 | 0.0032 | 112.208 |
| MLP (Backpropagation) | Deep Learning | 1.133 | 0.561 | 0.0112 | 119.528 |

### 5. Basic GBM Performance Evolution
**Historical performance improvements:**

| Date | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| May 19, 2025 (Early) | 17.295 | 8.839 | 0.3592 | Initial implementation |
| May 19, 2025 (Mid) | 11.538 | 6.759 | 0.3413 | Parameter tuning |
| May 19, 2025 (Late) | 7.312 | 4.208 | **0.7395** | Optimized features |
| May 20, 2025 | 2.370 | 1.022 | **0.9726** | Advanced preprocessing |

## Key Findings

### Best Overall Performance
1. **Wind-related predictions** show the highest accuracy:
   - Sunshine Duration: R² = 0.9824
   - Wind Direction: R² = 0.9778
   - Wind Speed: R² = 0.9461

2. **Temperature predictions** are highly accurate:
   - Average Temperature: R² = 0.9232

3. **Rainfall predictions** are more challenging but still good:
   - Rainfall: R² = 0.9007 (current best)

### Model Architecture Performance Ranking
1. **Gradient Boosting (GBM)**: Consistently best performer across all metrics
2. **Random Forest**: Good baseline performance, reliable
3. **SVM**: Moderate performance, stable
4. **Deep Learning Models** (LSTM, CNN, MLP): Generally underperformed for this dataset

### Forecast Horizon Analysis
- **Day 1 predictions**: Most accurate (R² ≈ 0.40)
- **Day 2-3 predictions**: Declining accuracy (R² ≈ 0.15-0.16)
- **Performance degradation**: ~60% accuracy loss from Day 1 to Day 3

### Parameter-Specific Insights
- **Sunshine Duration**: Most predictable weather parameter
- **Wind Direction**: High variance but good overall R²
- **Rainfall**: Most challenging due to high variability and sparse events
- **Temperature**: Consistent and highly predictable

## Technical Notes
- R² values closer to 1.0 indicate better model performance
- RMSE and MAE should be interpreted relative to the scale of each parameter
- MAPE values are extremely high for some models due to near-zero actual values in the dataset
- Multi-day forecasts show expected accuracy degradation with longer horizons

---
*Generated on: January 2025*
*Data covers training period: 1940-2025* 