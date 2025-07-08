# Perubahan Implementasi TimeSeriesSplit

## Ringkasan Perubahan

Kode telah diperbarui untuk mengganti K-Fold Cross Validation dengan TimeSeriesSplit yang lebih sesuai untuk data time series. Selain itu, telah ditambahkan visualisasi split data untuk semua target parameter dalam satu gambar.

## Perubahan Utama

### 1. Import Statement
```python
# Sebelum
from sklearn.model_selection import GridSearchCV, KFold

# Sesudah  
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
```

### 2. Konfigurasi Cross-Validation
```python
# Sebelum
tscv = KFold(n_splits=cv_folds, shuffle=False)

# Sesudah
tscv = TimeSeriesSplit(n_splits=cv_folds)
```

### 3. Metode Visualisasi Baru

Ditambahkan metode `plot_data_splits()` yang menghasilkan visualisasi komprehensif untuk semua target parameter:

#### Fitur Visualisasi:
- **Single Figure**: Semua target parameter ditampilkan dalam satu gambar
- **Shared X-axis**: Menggunakan timeline yang sama untuk semua target
- **Color-coded Splits**: Setiap split memiliki warna berbeda
- **Train/Validation Distinction**: Garis solid untuk training, garis putus-putus untuk validation
- **Statistics Box**: Informasi statistik split di setiap subplot

#### Struktur Plot:
```
┌─────────────────────────────────────┐
│ Rainfall (mm) - Time Series Split  │
│ [Split 1 Train] [Split 1 Val]     │
│ [Split 2 Train] [Split 2 Val]     │
│ ...                                │
├─────────────────────────────────────┤
│ Sunshine Duration (hours) - Split  │
│ [Split 1 Train] [Split 1 Val]     │
│ [Split 2 Train] [Split 2 Val]     │
│ ...                                │
├─────────────────────────────────────┤
│ Average Temperature (°C) - Split   │
│ [Split 1 Train] [Split 1 Val]     │
│ [Split 2 Train] [Split 2 Val]     │
│ ...                                │
├─────────────────────────────────────┤
│ Wind Direction - Split             │
│ [Split 1 Train] [Split 1 Val]     │
│ [Split 2 Train] [Split 2 Val]     │
│ ...                                │
├─────────────────────────────────────┤
│ Wind Speed (m/s) - Split          │
│ [Split 1 Train] [Split 1 Val]     │
│ [Split 2 Train] [Split 2 Val]     │
│ ...                                │
└─────────────────────────────────────┘
```

### 4. Keunggulan TimeSeriesSplit

#### Dibandingkan K-Fold:
1. **Temporal Ordering**: Menghormati urutan temporal data
2. **No Data Leakage**: Tidak ada informasi masa depan yang bocor ke training set
3. **Realistic Validation**: Simulasi kondisi prediksi yang lebih realistis
4. **Progressive Training**: Setiap split menggunakan data yang semakin banyak

#### Struktur Split:
```
Split 1: [Train: 0-20%] [Val: 20-40%]
Split 2: [Train: 0-40%] [Val: 40-60%]  
Split 3: [Train: 0-60%] [Val: 60-80%]
Split 4: [Train: 0-80%] [Val: 80-100%]
Split 5: [Train: 0-100%] [Val: -]
```

### 5. Output Visualisasi

#### File Output:
- `data_splits_visualization.png`: Visualisasi utama split data
- Lokasi: `gbm/plots/[timestamp]/data_splits_visualization.png`

#### Informasi yang Ditampilkan:
- Timeline lengkap untuk semua target
- Warna berbeda untuk setiap split
- Legend yang menjelaskan train/validation
- Statistik split di setiap subplot
- Grid untuk memudahkan pembacaan

### 6. Cara Menjalankan

#### Opsi 1: Jalankan Script Test
```bash
python test_timeseries_split.py
```

#### Opsi 2: Jalankan Main Script
```bash
python final_main_gbm.py
```

### 7. Interpretasi Visualisasi

#### Membaca Plot:
1. **Timeline**: Sumbu X menunjukkan waktu (tanggal)
2. **Values**: Sumbu Y menunjukkan nilai variabel cuaca
3. **Colors**: Setiap warna mewakili satu split
4. **Line Styles**: 
   - Solid line = Training data
   - Dashed line = Validation data
5. **Progression**: Split berikutnya menggunakan data training yang lebih banyak

#### Analisis Split:
- **Split 1**: Training dengan data awal, validation dengan data tengah
- **Split 2**: Training dengan data lebih banyak, validation dengan data lebih lanjut
- **Split 3-5**: Progresif menggunakan data training yang semakin banyak

### 8. Keuntungan Implementasi Ini

1. **Visualisasi Komprehensif**: Semua target dalam satu gambar
2. **Timeline Konsisten**: X-axis yang sama untuk semua target
3. **Informasi Lengkap**: Statistik dan legend yang informatif
4. **Kualitas Tinggi**: Output PNG dengan DPI 300
5. **Debugging Friendly**: Print statements untuk monitoring

### 9. Monitoring dan Debugging

#### Console Output:
```
Time Series Split Statistics:
Total samples: [jumlah]
Number of splits: 5
Split 1: Train=[jumlah] samples, Val=[jumlah] samples
Split 2: Train=[jumlah] samples, Val=[jumlah] samples
...
```

#### File Output:
- Plot tersimpan di direktori timestamp
- Nama file: `data_splits_visualization.png`
- Format: High-resolution PNG

Implementasi ini memberikan pemahaman yang lebih baik tentang bagaimana data dibagi untuk training dan validation dalam konteks time series, serta memastikan tidak ada data leakage yang dapat mempengaruhi validitas model. 