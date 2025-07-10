# CSV Output Documentation - Final Main GBM

Dokumentasi ini menjelaskan file-file CSV yang dihasilkan oleh sistem prediksi cuaca GBM (`final_main_gbm.py`).

## Lokasi Output
File CSV disimpan di direktori: `gbm/csv_outputs/{timestamp}/`

Dimana `{timestamp}` adalah waktu eksekusi dalam format `YYYYMMDD_HHMMSS`.

## File-file CSV yang Dihasilkan

### 1. Feature Importance All Targets (`feature_importance_all_targets.csv`)

**Deskripsi**: Tabel importance fitur untuk semua target variabel secara keseluruhan.

**Kolom:**
- `Rank`: Peringkat fitur berdasarkan average importance
- `Feature_Name`: Nama fitur
- `Feature_Type`: Tipe fitur (Original/Engineered)
- `Average_Importance`: Rata-rata importance across all targets
- `Std_Importance`: Standard deviasi importance
- `{target}_Importance`: Importance untuk setiap target (RR, ss, Tavg, ddd_car, ff_avg)

**Contoh Isi:**
```csv
Rank,Feature_Name,Feature_Type,Average_Importance,Std_Importance,RR_Importance
1,Heat_Index,Engineered,0.163402,0.0,0.163402
2,Temp_Humidity,Engineered,0.144179,0.0,0.144179
3,Dew_Point,Engineered,0.13881,0.0,0.13881
```

**Kegunaan:**
- Analisis fitur mana yang paling berpengaruh terhadap prediksi
- Validasi efektivitas engineered features
- Feature selection untuk model optimization

### 2. Prediction Samples (`prediction_samples_{n}.csv`)

**Deskripsi**: Tabel perhitungan prediksi untuk n data pertama (default: 10 samples) beserta tanggalnya.

**Kolom:**
- `Sample_ID`: ID sample dari dataset
- `Date`: Tanggal dalam format YYYY-MM-DD
- `Day_of_Week`: Hari dalam seminggu
- `Month`: Nama bulan
- `{target}_Actual`: Nilai aktual untuk setiap target
- `{target}_Predicted`: Nilai prediksi untuk setiap target
- `{target}_Absolute_Error`: Error absolut
- `{target}_Percentage_Error`: Error dalam persen

**Contoh Isi:**
```csv
Sample_ID,Date,Day_of_Week,Month,RR_Actual,RR_Predicted,RR_Absolute_Error,RR_Percentage_Error
80,2010-03-22,Monday,March,35.0,36.6777,1.6777,4.79
81,2010-03-23,Tuesday,March,35.0,27.7595,7.2405,20.69
```

**Kegunaan:**
- Validasi kualitas prediksi pada sample tertentu
- Analisis pola error berdasarkan waktu
- Debugging model performance

### 3. Accuracy Metrics All Models (`accuracy_metrics_all_models.csv`)

**Deskripsi**: Tabel metrik akurasi untuk semua model (single-day dan multi-day forecasting).

**Kolom:**
- `Target_Variable`: Kode target variable (RR, ss, Tavg, ddd_car, ff_avg)
- `Target_Name`: Nama lengkap target variable
- `Model_Type`: Tipe model (Single_Day/Multi_Day)
- `Forecast_Horizon`: Horizon prediksi (0=hari ini, 1-5=hari ke-depan)
- `Forecast_Label`: Label horizon (Today, 1_Day_Ahead, dll)
- `MSE`: Mean Squared Error
- `RMSE`: Root Mean Squared Error
- `MAE`: Mean Absolute Error
- `R2_Score`: R-squared score
- `Model_Algorithm`: Algoritma yang digunakan
- `Target_Unit`: Unit target variable
- `Target_Range`: Range nilai target variable

**Contoh Isi:**
```csv
Target_Variable,Target_Name,Model_Type,Forecast_Horizon,Forecast_Label,MSE,RMSE,MAE,R2_Score,Model_Algorithm,Target_Unit,Target_Range
RR,Rainfall (mm),Single_Day,0,Today,456.561396,21.3673,18.3135,-0.3008,GradientBoostingRegressor,mm,"[0, ∞)"
RR,Rainfall (mm),Multi_Day,1,1_Day_Ahead,523.245,22.8747,19.2156,-0.1523,GradientBoostingRegressor,mm,"[0, ∞)"
```

**Kegunaan:**
- Evaluasi performance model untuk setiap target
- Perbandingan akurasi antar target variables
- Analisis degradasi akurasi dengan forecast horizon

### 4. Accuracy Metrics Summary (`accuracy_metrics_summary.csv`)

**Deskripsi**: Ringkasan statistik metrik akurasi per target variable.

**Kolom:**
- `Target_Variable`: Kode target variable
- `Target_Name`: Nama lengkap target variable
- `Number_of_Models`: Jumlah model untuk target ini
- `Best_R2_Score`: R² terbaik
- `Worst_R2_Score`: R² terburuk
- `Average_R2_Score`: Rata-rata R²
- `Best_RMSE`: RMSE terbaik (terkecil)
- `Worst_RMSE`: RMSE terburuk (terbesar)
- `Average_RMSE`: Rata-rata RMSE

**Contoh Isi:**
```csv
Target_Variable,Target_Name,Number_of_Models,Best_R2_Score,Worst_R2_Score,Average_R2_Score,Best_RMSE,Worst_RMSE,Average_RMSE
RR,Rainfall (mm),6,0.3901,0.1234,0.2567,16.8728,25.4321,21.1524
ss,Sunshine Duration (hours),6,0.4306,0.2145,0.3225,2.4989,3.8765,3.1877
```

**Kegunaan:**
- Overview performance model secara keseluruhan
- Identifikasi target variable yang sulit diprediksi
- Benchmark untuk improvement model

## Target Variables

1. **RR**: Rainfall (mm) - Curah hujan dalam milimeter
2. **ss**: Sunshine Duration (hours) - Durasi sinar matahari dalam jam
3. **Tavg**: Average Temperature (°C) - Suhu rata-rata dalam Celsius
4. **ddd_car**: Wind Direction (degrees) - Arah angin dalam derajat
5. **ff_avg**: Wind Speed (m/s) - Kecepatan angin dalam meter per detik

## Feature Types

1. **Original**: Fitur dari dataset asli
2. **Engineered**: Fitur yang dibuat melalui feature engineering:
   - `Temp_Range`: Selisih suhu maksimum dan minimum
   - `Temp_Humidity`: Interaksi suhu dan kelembaban
   - `Temp_Range_RH`: Interaksi range suhu dan kelembaban
   - `Dew_Point`: Titik embun (dihitung)
   - `Heat_Index`: Indeks panas
   - `Rain_Streak`: Streak hari hujan berturut-turut
   - `Dry_Streak`: Streak hari kering berturut-turut
   - `Month_sin`, `Month_cos`: Encoding siklik bulan
   - `Day_sin`, `Day_cos`: Encoding siklik hari
   - `DayOfYear_sin`, `DayOfYear_cos`: Encoding siklik hari dalam tahun

## Cara Menggunakan

1. **Jalankan model:**
   ```bash
   python final_main_gbm.py
   ```

2. **Lokasi output:**
   File CSV akan tersimpan di `gbm/csv_outputs/{timestamp}/`

3. **Import untuk analisis:**
   ```python
   import pandas as pd
   
   # Load feature importance
   feature_imp = pd.read_csv('gbm/csv_outputs/{timestamp}/feature_importance_all_targets.csv')
   
   # Load predictions
   predictions = pd.read_csv('gbm/csv_outputs/{timestamp}/prediction_samples_10.csv')
   
   # Load metrics
   metrics = pd.read_csv('gbm/csv_outputs/{timestamp}/accuracy_metrics_all_models.csv')
   ```

## Interpretasi Hasil

### Feature Importance
- Nilai importance menunjukkan kontribusi relatif fitur terhadap prediksi
- Engineered features yang ranking tinggi menunjukkan keberhasilan feature engineering
- Standard deviasi tinggi menunjukkan fitur memiliki importance yang berbeda antar target

### Prediction Accuracy
- R² > 0.7: Prediksi sangat baik
- R² 0.4-0.7: Prediksi baik
- R² 0.2-0.4: Prediksi cukup
- R² < 0.2: Prediksi perlu perbaikan

### Error Analysis
- MAE memberikan gambaran error rata-rata dalam unit asli
- RMSE lebih sensitif terhadap outlier
- Percentage error berguna untuk memahami error relatif

## Troubleshooting

1. **File tidak terbuat**: Pastikan direktori `gbm/csv_outputs/` memiliki permission write
2. **Error dalam CSV**: Periksa format data input dan pastikan tidak ada missing values
3. **Performance buruk**: Pertimbangkan feature selection atau hyperparameter tuning 