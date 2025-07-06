# IMAKAS Algorithm - Mathematical Equations Reference

## 1. Feature Engineering Equations

### 1.1 Cyclical Encoding
For temporal features with periodic nature:

**Hour Encoding (24-hour cycle):**
```
Hour_sin = sin(2π × Hour / 24)
Hour_cos = cos(2π × Hour / 24)
```

**Day Encoding (31-day cycle):**
```
Day_sin = sin(2π × Day / 31)
Day_cos = cos(2π × Day / 31)
```

**Month Encoding (12-month cycle):**
```
Month_sin = sin(2π × Month / 12)
Month_cos = cos(2π × Month / 12)
```

**General Cyclical Encoding:**
```
Feature_sin = sin(2π × Feature / Period)
Feature_cos = cos(2π × Feature / Period)
```

### 1.2 Meteorological Derived Features

**Temperature-Dew Point Difference:**
```
Temp_Dew_Diff = Temperature_C - Dew_Point_C
```

**Wind-Pressure Ratio:**
```
Wind_Pressure_Ratio = Speed_kmh / (Pressure_hPa / 1000)
```

**Simplified Vapor Pressure Deficit:**
```
VPD = Temp_Dew_Diff × 0.1
```

**Season Calculation:**
```
Season = floor((Month % 12 + 3) / 3)
```
Where: Winter=1, Spring=2, Summer=3, Fall=4

### 1.3 Rolling Window Statistics

**Rolling Mean (n-period window):**
```
Rolling_Mean(t) = (1/n) × Σ(i=t-n+1 to t) X(i)
```

**Rolling Standard Deviation:**
```
Rolling_Std(t) = √[(1/n) × Σ(i=t-n+1 to t) (X(i) - Rolling_Mean(t))²]
```

**Rolling Variance:**
```
Rolling_Var(t) = (1/n) × Σ(i=t-n+1 to t) (X(i) - Rolling_Mean(t))²
```

### 1.4 Lag Features

**General Lag Feature:**
```
X_lag_k(t) = X(t-k)
```

**Specific Lag Transformations:**
```
Variable_Lag_1h = Variable.shift(2)   # 2 records = 1 hour
Variable_Lag_3h = Variable.shift(6)   # 6 records = 3 hours
Variable_Lag_6h = Variable.shift(12)  # 12 records = 6 hours
```

## 2. Data Preprocessing Equations

### 2.1 Standard Scaling (Z-score normalization)

**Feature Scaling:**
```
X_scaled = (X - μ) / σ

Where:
μ = (1/n) × Σ(i=1 to n) x_i    (sample mean)
σ = √[(1/n) × Σ(i=1 to n) (x_i - μ)²]    (sample standard deviation)
```

**Inverse Scaling:**
```
X_original = X_scaled × σ + μ
```

### 2.2 Target Shifting for Multi-Horizon Forecasting

**h-step ahead target:**
```
Y_future_h(t) = Y(t + h)
```

**For 30-minute intervals:**
```
steps = hours × 2
Y_future_hours = Y.shift(-steps)
```

## 3. Gradient Boosting Mathematics

### 3.1 Objective Function

**General Loss Function:**
```
L(y, F(x)) = Σ(i=1 to n) l(y_i, F(x_i)) + Ω(F)
```

Where:
- `l(y_i, F(x_i))` is the loss function (MSE for regression)
- `Ω(F)` is the regularization term
- `F(x)` is the ensemble prediction

**Mean Squared Error Loss:**
```
l(y_i, F(x_i)) = (1/2) × (y_i - F(x_i))²
```

### 3.2 Gradient Boosting Algorithm

**Initialization:**
```
F_0(x) = argmin_γ Σ(i=1 to n) l(y_i, γ)
```

**For regression with MSE:**
```
F_0(x) = ȳ = (1/n) × Σ(i=1 to n) y_i
```

**Iterative Updates (m = 1 to M):**

**Step 1: Compute Negative Gradients**
```
r_im = -[∂l(y_i, F(x_i))/∂F(x_i)]|_{F=F_{m-1}}
```

**For MSE loss:**
```
r_im = y_i - F_{m-1}(x_i)
```

**Step 2: Fit Regression Tree**
```
{R_jm}_{j=1}^{J_m} = argmin_R Σ(i=1 to n) (r_im - R(x_i))²
```

**Step 3: Compute Terminal Node Values**
```
γ_jm = argmin_γ Σ_{x_i ∈ R_jm} l(y_i, F_{m-1}(x_i) + γ)
```

**For MSE loss:**
```
γ_jm = (1/|R_jm|) × Σ_{x_i ∈ R_jm} r_im
```

**Step 4: Update Model**
```
F_m(x) = F_{m-1}(x) + ν × Σ(j=1 to J_m) γ_jm × I(x ∈ R_jm)
```

Where `ν` is the learning rate and `I(·)` is the indicator function.

**Final Model:**
```
F(x) = F_M(x) = F_0(x) + Σ(m=1 to M) ν × Σ(j=1 to J_m) γ_jm × I(x ∈ R_jm)
```

### 3.3 Regularization Terms

**L1 Regularization (Lasso):**
```
Ω(F) = λ × Σ(j=1 to p) |w_j|
```

**L2 Regularization (Ridge):**
```
Ω(F) = λ × Σ(j=1 to p) w_j²
```

**Tree Complexity Regularization:**
```
Ω(f_m) = γ × T_m + (1/2) × λ × Σ(j=1 to T_m) w_j²
```

Where `T_m` is the number of leaves in tree m.

## 4. Evaluation Metrics

### 4.1 Regression Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(i=1 to n) (y_i - ŷ_i)²
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE = √[(1/n) × Σ(i=1 to n) (y_i - ŷ_i)²]
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ(i=1 to n) |y_i - ŷ_i|
```

**R-squared (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ(i=1 to n) (y_i - ŷ_i)²     (Residual sum of squares)
SS_tot = Σ(i=1 to n) (y_i - ȳ)²       (Total sum of squares)
ȳ = (1/n) × Σ(i=1 to n) y_i           (Mean of observed values)
```

**Adjusted R-squared:**
```
R²_adj = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]
```

Where `p` is the number of predictors.

### 4.2 Cross-Validation Metrics

**K-Fold Cross-Validation Score:**
```
CV_Score = (1/K) × Σ(k=1 to K) Score_k
```

**Cross-Validation Standard Deviation:**
```
CV_Std = √[(1/K) × Σ(k=1 to K) (Score_k - CV_Score)²]
```

**Cross-Validation Standard Error:**
```
CV_SE = CV_Std / √K
```

## 5. SHAP (Shapley Values) Mathematics

### 5.1 Shapley Value Formula

**General Shapley Value:**
```
φ_i = Σ_{S⊆N\{i}} [|S|! × (|N| - |S| - 1)! / |N|!] × [f(S ∪ {i}) - f(S)]
```

Where:
- `φ_i` is the SHAP value for feature i
- `N` is the set of all features
- `S` is a subset of features not including i
- `f(S)` is the model prediction using feature subset S

### 5.2 Efficiency Property

**Additivity:**
```
f(x) = E[f(X)] + Σ(i=1 to p) φ_i(x)
```

Where `E[f(X)]` is the expected model output (baseline).

### 5.3 TreeSHAP Algorithm

**For tree-based models, efficient computation:**
```
φ_i = Σ_{T} w_T × (f_T(x_{S∪{i}}) - f_T(x_S))
```

Where `T` represents all possible paths through the tree.

## 6. Physical Constraint Equations

### 6.1 Post-Prediction Constraints

**Humidity Constraint:**
```
Humidity_constrained = clip(Humidity_predicted, 0, 100)
```

**Precipitation Constraints:**
```
Precip_Rate_constrained = max(Precip_Rate_predicted, 0)
Precip_Accum_constrained = max(Precip_Accum_predicted, 0)
```

**Solar Radiation Constraint:**
```
Solar_constrained = max(Solar_predicted, 0)
```

**Temperature Constraint (optional):**
```
Temperature_constrained = clip(Temperature_predicted, -50, 60)  # Reasonable range for Earth
```

### 6.2 Physical Relationship Checks

**Dew Point Constraint:**
```
Dew_Point ≤ Temperature_C  (always true in nature)
```

**Relative Humidity Relationship:**
```
RH = (e / e_s) × 100%

Where:
e = actual vapor pressure
e_s = saturation vapor pressure at temperature T
```

**Simplified Saturation Vapor Pressure (Magnus formula):**
```
e_s = 6.112 × exp[(17.67 × T) / (T + 243.5)]
```

Where T is temperature in Celsius and e_s is in hPa.

## 7. Time Series Specific Equations

### 7.1 Temporal Split

**Training Set:**
```
X_train = X[0 : floor(0.8 × n)]
y_train = y[0 : floor(0.8 × n)]
```

**Testing Set:**
```
X_test = X[floor(0.8 × n) : n]
y_test = y[floor(0.8 × n) : n]
```

### 7.2 Forecast Horizon Accuracy Degradation

**Empirical Relationship:**
```
R²(h) = R²(1) × exp(-α × h)
```

Where:
- `R²(h)` is R-squared at horizon h
- `R²(1)` is R-squared at 1-hour horizon
- `α` is the decay parameter (target-specific)
- `h` is the forecast horizon in hours

### 7.3 Seasonal Decomposition

**Additive Model:**
```
X(t) = Trend(t) + Seasonal(t) + Residual(t)
```

**Multiplicative Model:**
```
X(t) = Trend(t) × Seasonal(t) × Residual(t)
```

## 8. Computational Complexity

### 8.1 Training Time Complexity

**Gradient Boosting:**
```
O(n × m × d × T)
```

Where:
- `n` = number of samples
- `m` = number of features
- `d` = maximum tree depth
- `T` = number of trees

### 8.2 Memory Complexity

**Feature Matrix Storage:**
```
Memory = n × m × sizeof(float64) = n × m × 8 bytes
```

**Model Storage:**
```
Model_Size ≈ T × (2^d - 1) × sizeof(node) × number_of_targets
```

### 8.3 Prediction Time Complexity

**Single Prediction:**
```
O(T × d × number_of_targets)
```

**Batch Prediction:**
```
O(n_pred × T × d × number_of_targets)
```

## 9. Statistical Significance Tests

### 9.1 Model Comparison

**Paired t-test for model comparison:**
```
t = (d̄ - μ_0) / (s_d / √n)
```

Where:
- `d̄` is the mean difference in performance
- `s_d` is the standard deviation of differences
- `n` is the number of comparisons

### 9.2 Feature Importance Significance

**Permutation Importance Standard Error:**
```
SE = √[(1/n) × Σ(i=1 to n) (Imp_i - Imp_mean)²]
```

**Confidence Interval:**
```
CI = Imp_mean ± (t_α/2 × SE)
```

Where `t_α/2` is the critical t-value for desired confidence level.

## 10. Optimization Equations

### 10.1 Hyperparameter Optimization

**Grid Search Objective:**
```
θ* = argmin_θ (1/K) × Σ(k=1 to K) L(y_k, f_θ(X_k))
```

**Bayesian Optimization:**
```
θ_next = argmax_θ α(θ|D_1:t)
```

Where `α` is the acquisition function and `D_1:t` is the observed data.

### 10.2 Early Stopping Criterion

**Validation Loss Improvement:**
```
Stop if: L_val(t) > L_val(t-patience) - min_delta
```

Where `patience` is the number of epochs to wait and `min_delta` is the minimum improvement threshold.

---

## Summary

These mathematical equations form the foundation of the IMAKAS Weather Prediction Algorithm. The system combines:

1. **Advanced feature engineering** with cyclical encoding and meteorological physics
2. **Gradient boosting mathematics** for ensemble learning
3. **Multi-target, multi-horizon** prediction capabilities
4. **Statistical validation** through cross-validation and significance testing
5. **Explainable AI** through SHAP value computation
6. **Physical constraints** to ensure realistic predictions

The mathematical rigor ensures both accuracy and interpretability, making the system suitable for operational weather forecasting and research applications. 