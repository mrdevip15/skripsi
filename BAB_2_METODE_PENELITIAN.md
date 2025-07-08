# BAB 2 METODE PENELITIAN

## 2.1 Desain Penelitian

Penelitian ini menggunakan pendekatan kuantitatif dengan metode eksperimental untuk mengembangkan sistem prediksi cuaca multi-target menggunakan algoritma Gradient Boosting Machine (GBM). Desain penelitian dirancang untuk memprediksi lima variabel cuaca utama secara simultan dengan horizon prediksi multi-hari.

### 2.1.1 Variabel Penelitian

**Variabel Target (Dependent Variables):**
1. **RR** - Curah Hujan (mm)
2. **ss** - Durasi Sinar Matahari (jam)
3. **Tavg** - Suhu Rata-rata (°C)
4. **ddd_car** - Arah Angin (derajat)
5. **ff_avg** - Kecepatan Angin (m/s)

**Variabel Prediktor (Independent Variables):**
- Variabel cuaca dasar: Suhu minimum (Tn), suhu maksimum (Tx), kelembaban relatif (RH_avg)
- Variabel temporal: Encoding siklis bulan (Month_sin, Month_cos)
- Variabel turunan: Rentang suhu (Temp_Range), titik embun (Dew_Point)
- Variabel historis: Moving average dan lag features
- Variabel pola: Streak hujan dan kering

## 2.2 Sumber Data

### 2.2.1 Dataset
Dataset yang digunakan adalah data cuaca historis Makassar yang tersimpan dalam file `makassar.csv`. Dataset ini berisi pengamatan harian parameter meteorologi dengan struktur time series.

### 2.2.2 Karakteristik Data
- **Format**: Data time series harian
- **Periode**: Data historis cuaca Makassar
- **Variabel**: 5 variabel target dan multiple variabel prediktor
- **Missing Values**: Ditangani dengan nilai khusus (8888, 9999) yang memerlukan interpolasi

## 2.3 Preprocessing Data

### 2.3.1 Pembersihan Data (Data Cleaning)

**Penanganan Nilai Khusus:**
```python
# Identifikasi nilai khusus (8888, 9999)
special_values = [8888, 9999]
```

**Strategi Interpolasi Multi-Tahap:**
1. **Forward Fill**: Propagasi nilai valid terakhir ke depan
2. **Backward Fill**: Propagasi nilai valid berikutnya ke belakang
3. **5-Day Rolling Mean**: Rata-rata bergerak 5 hari untuk nilai yang tersisa
4. **Median Imputation**: Penggunaan median kolom sebagai fallback terakhir

### 2.3.2 Feature Engineering

**Variabel Temporal:**
```python
# Encoding siklis untuk menangkap pola musiman
df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
```

**Variabel Meteorologi Turunan:**
```python
# Rentang suhu
df['Temp_Range'] = df['Tx'] - df['Tn']

# Titik embun (simplified)
df['Dew_Point'] = temp_avg_calc - ((100 - df['RH_avg']) / 5)

# Heat index
df['Heat_Index'] = temp_avg_calc + 0.05 * df['RH_avg']
```

**Variabel Interaksi:**
```python
# Interaksi suhu-kelembaban
df['Temp_Humidity'] = ((df['Tx'] + df['Tn']) / 2) * df['RH_avg']
df['Temp_Range_RH'] = df['Temp_Range'] * df['RH_avg']
```

**Rolling Features:**
```python
# Moving averages dengan window 3, 7, dan 14 hari
windows = [3, 7, 14]
for window in windows:
    for target in self.target_columns:
        df[f'{target}_Rolling_Mean_{window}d'] = df[target].rolling(window=window, min_periods=1).mean()
        df[f'{target}_Rolling_Std_{window}d'] = df[target].rolling(window=window, min_periods=1).std()
```

**Lag Features:**
```python
# Lag features untuk menangkap dependensi temporal
for lag in [1, 2, 3, 5, 7, 14]:
    for target in self.target_columns:
        df[f'{target}_Lag_{lag}'] = df[target].shift(lag)
```

**Pattern Features:**
```python
# Streak patterns untuk hujan
df['Rain_Streak'] = consecutive_rain_days
df['Dry_Streak'] = consecutive_dry_days
```

### 2.3.3 Normalisasi Data

**Feature Scaling:**
```python
# StandardScaler untuk normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Target Scaling:**
```python
# Scaling khusus untuk target tertentu
if target in ['RR', 'ss', 'ff_avg']:
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
```

## 2.4 Arsitektur Model

### 2.4.1 Multi-Target Gradient Boosting Machine

**Konsep Dasar:**
Penelitian ini mengimplementasikan arsitektur multi-target dimana setiap variabel cuaca memiliki model GBM terpisah yang dioptimalkan untuk karakteristik spesifik masing-masing variabel.

**Konfigurasi Model per Target:**

1. **Curah Hujan (RR):**
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=4,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

2. **Durasi Sinar Matahari (ss):**
```python
GradientBoostingRegressor(
    n_estimators=180,
    learning_rate=0.06,
    max_depth=4,
    min_samples_split=6,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

3. **Suhu Rata-rata (Tavg):**
```python
GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    min_samples_split=8,
    min_samples_leaf=6,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

4. **Arah Angin (ddd_car):**
```python
GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    min_samples_split=8,
    min_samples_leaf=6,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

5. **Kecepatan Angin (ff_avg):**
```python
GradientBoostingRegressor(
    n_estimators=160,
    learning_rate=0.07,
    max_depth=4,
    min_samples_split=7,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

### 2.4.2 Multi-Day Forecasting

**Strategi Prediksi Multi-Hari:**
```python
# Persiapan dataset untuk prediksi multi-hari
for day in range(1, self.forecast_days + 1):
    day_df[f'Future_{target}_{day}d'] = df[target].shift(-day)
```

**Horizon Prediksi:**
- Day 0: Prediksi hari ini
- Day 1-5: Prediksi 1-5 hari ke depan

## 2.5 Metodologi Pelatihan

### 2.5.1 Pembagian Data

**Temporal Split:**
```python
# Pembagian data dengan mempertahankan urutan temporal
train_size = int(0.8 * len(df))
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
```

**Rasio Pembagian:**
- Training: 80% data awal
- Testing: 20% data akhir

### 2.5.2 Cross-Validation

**Time Series Cross-Validation:**
```python
# K-Fold CV tanpa shuffling untuk data time series
tscv = KFold(n_splits=cv_folds, shuffle=False)
```

**Strategi CV:**
- 3-fold CV untuk prediksi hari ini (day 0)
- 2-fold CV untuk prediksi multi-hari (day 1-5)

### 2.5.3 Ensemble Prediction

**Averaging Across Folds:**
```python
# Rata-rata prediksi dari semua fold
final_predictions = np.mean(fold_predictions, axis=0)
```

**Uncertainty Estimation:**
```python
# Estimasi ketidakpastian dari standar deviasi prediksi
pred_std = np.std(fold_predictions, axis=0)
```

## 2.6 Constraint Application

### 2.6.1 Physical Constraints

**Curah Hujan:**
```python
# Curah hujan tidak boleh negatif
y_pred = np.maximum(y_pred, 0)
```

**Durasi Sinar Matahari:**
```python
# Durasi sinar matahari: 0-24 jam
y_pred = np.clip(y_pred, 0, 24)
```

**Kecepatan Angin:**
```python
# Kecepatan angin tidak boleh negatif
y_pred = np.maximum(y_pred, 0)
```

**Arah Angin:**
```python
# Arah angin: 0-360 derajat (circular)
y_pred = np.clip(y_pred % 360, 0, 360)
```

## 2.7 Evaluasi Model

### 2.7.1 Metrik Evaluasi

**Metrik Regresi:**
```python
metrics = {
    'mse': mean_squared_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
```

**Metrik Khusus Curah Hujan:**
```python
# Akurasi deteksi hujan (binary classification)
rain_detection_accuracy = np.mean((y_test > 0) == (y_pred > 0))
```

### 2.7.2 Visualisasi

**Plot Prediksi vs Aktual:**
- Time series plot untuk setiap target
- Scatter plot actual vs predicted
- Residual analysis

**Feature Importance:**
- Ranking fitur berdasarkan importance score
- Correlation matrix
- Feature distribution analysis

**Multi-Day Performance:**
- Heatmap performa R² untuk semua target dan horizon
- Line plot degradasi performa vs horizon prediksi

## 2.8 Implementasi Sistem

### 2.8.1 Struktur Kode

**Class Architecture:**
```python
class GBMWeatherPredictor:
    def __init__(self):
        self.multi_day_models = {}
        self.multi_target_models = {}
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
```

**Pipeline Utama:**
1. `preprocess_data()` - Preprocessing dan feature engineering
2. `train_and_evaluate()` - Pelatihan model multi-target
3. `prepare_multi_day_dataset()` - Persiapan data multi-hari
4. `train_multi_day_models()` - Pelatihan model multi-hari
5. `save_multi_target_models()` - Penyimpanan model

### 2.8.2 Optimisasi Komputasi

**Parallel Processing:**
```python
# Konfigurasi CPU untuk parallel processing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)
```

**Memory Management:**
```python
# Non-interactive backend untuk matplotlib
matplotlib.use('Agg')
```

## 2.9 Validasi dan Verifikasi

### 2.9.1 Validasi Model

**Cross-Validation Strategy:**
- Time series split untuk menghindari data leakage
- Multiple fold validation untuk robustness
- Separate validation untuk setiap target

**Performance Monitoring:**
- Tracking CV scores across folds
- Monitoring training convergence
- Validation loss tracking

### 2.9.2 Verifikasi Hasil

**Constraint Verification:**
- Pemeriksaan physical constraints
- Validation range nilai prediksi
- Consistency check antar target

**Statistical Validation:**
- Significance testing
- Confidence interval calculation
- Residual analysis

## 2.10 Penyimpanan dan Deployment

### 2.10.1 Model Serialization

**Model Storage:**
```python
# Penyimpanan model dengan joblib
joblib.dump(model_data['model'], model_path)
```

**Metadata Storage:**
```python
# Penyimpanan feature list dan scaler
with open(features_path, 'wb') as f:
    pickle.dump(model_data['features_used'], f)
```

### 2.10.2 Directory Structure

```
gbm/
├── models/
│   ├── target_RR/
│   │   ├── day_0/
│   │   ├── day_1/
│   │   └── ...
│   ├── target_ss/
│   └── ...
├── plots/
│   └── [timestamp]/
└── logs/
```

Metodologi ini dirancang untuk menghasilkan sistem prediksi cuaca yang robust, scalable, dan dapat diandalkan dengan kemampuan prediksi multi-target dan multi-hari yang komprehensif. 