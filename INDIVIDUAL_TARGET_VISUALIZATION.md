# Visualisasi Split Data Per Target Parameter

## ✅ Implementasi Berhasil

Visualisasi telah diperbarui untuk membuat gambar terpisah untuk setiap parameter target dengan split yang disusun ke bawah.

## 📊 File Output yang Dihasilkan

### 1. File Visualisasi Per Target:
- ✅ `data_splits_rr.png` (1.4MB) - Curah Hujan (mm)
- ✅ `data_splits_ss.png` (2.2MB) - Durasi Sinar Matahari (jam)
- ✅ `data_splits_tavg.png` (2.0MB) - Suhu Rata-rata (°C)
- ✅ `data_splits_ddd_car.png` (1.8MB) - Arah Angin (derajat)
- ✅ `data_splits_ff_avg.png` (1.3MB) - Kecepatan Angin (m/s)

### 2. Lokasi File:
```
gbm/plots/20250708_102027/
├── data_splits_rr.png
├── data_splits_ss.png
├── data_splits_tavg.png
├── data_splits_ddd_car.png
├── data_splits_ff_avg.png
└── original_target_variables_distribution.png
```

## 🎨 Struktur Visualisasi Baru

### Layout Per Gambar:
```
┌─────────────────────────────────────────────────────────┐
│              [Target Name] - Time Series Split Analysis │
│              Total samples: 5295, Number of splits: 5   │
├─────────────────────────────────────────────────────────┤
│ Split 1: [Target Name] - Split 1                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Train: 885 samples, Val: 882 samples              │ │
│ │ Train period: 2000-01-01 to 2002-06-30            │ │
│ │ Val period: 2002-07-01 to 2004-12-31              │ │
│ │ [Timeline with Train/Val data]                    │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Split 2: [Target Name] - Split 2                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Train: 1767 samples, Val: 882 samples             │ │
│ │ Train period: 2000-01-01 to 2004-12-31            │ │
│ │ Val period: 2005-01-01 to 2007-06-30              │ │
│ │ [Timeline with Train/Val data]                    │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Split 3: [Target Name] - Split 3                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Train: 2649 samples, Val: 882 samples             │ │
│ │ Train period: 2000-01-01 to 2007-06-30            │ │
│ │ Val period: 2007-07-01 to 2009-12-31              │ │
│ │ [Timeline with Train/Val data]                    │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Split 4: [Target Name] - Split 4                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Train: 3531 samples, Val: 882 samples             │ │
│ │ Train period: 2000-01-01 to 2009-12-31            │ │
│ │ Val period: 2010-01-01 to 2012-06-30              │ │
│ │ [Timeline with Train/Val data]                    │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Split 5: [Target Name] - Split 5                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Train: 4413 samples, Val: 882 samples             │ │
│ │ Train period: 2000-01-01 to 2012-06-30            │ │
│ │ Val period: 2012-07-01 to 2014-12-31              │ │
│ │ [Timeline with Train/Val data]                    │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🔍 Fitur Visualisasi Per Target

### 1. **Individual Figures**
- Setiap target parameter memiliki gambar terpisah
- Ukuran gambar: 15 x (4 × 5) = 15 x 20 inches
- Resolusi tinggi: DPI 300

### 2. **Vertical Split Arrangement**
- Setiap split ditampilkan dalam subplot terpisah
- Susunan vertikal dari atas ke bawah
- Split 1 di atas, Split 5 di bawah

### 3. **Enhanced Information**
- **Title**: Nama target dan nomor split
- **Statistics Box**: 
  - Jumlah samples train/validation
  - Periode waktu train/validation
  - Format tanggal yang jelas
- **Legend**: Train (solid line) vs Validation (dashed line)
- **Grid**: Untuk memudahkan pembacaan

### 4. **Color Coding**
- Setiap split memiliki warna berbeda
- Warna konsisten di semua target
- Background data dalam warna abu-abu transparan

## 📈 Analisis Per Target

### 1. **Curah Hujan (RR)**
- Range: 0-218.80 mm
- Karakteristik: Zero-inflated, sporadic peaks
- Visualisasi: Menunjukkan pola hujan yang tidak teratur

### 2. **Durasi Sinar Matahari (ss)**
- Range: 0-11.70 jam
- Karakteristik: Seasonal patterns, daily variation
- Visualisasi: Menunjukkan pola musiman yang jelas

### 3. **Suhu Rata-rata (Tavg)**
- Range: 23.40-33.20°C
- Karakteristik: Relatively stable, slight seasonal variation
- Visualisasi: Menunjukkan stabilitas dengan variasi musiman

### 4. **Arah Angin (ddd_car)**
- Range: 0-315 derajat
- Karakteristik: Circular data, directional patterns
- Visualisasi: Menunjukkan pola arah angin yang beragam

### 5. **Kecepatan Angin (ff_avg)**
- Range: 0-10 m/s
- Karakteristik: Low values, occasional peaks
- Visualisasi: Menunjukkan kecepatan angin yang relatif rendah

## 🎯 Keunggulan Visualisasi Baru

### 1. **Detail Per Target**
- Fokus pada satu parameter per gambar
- Analisis detail untuk setiap target
- Tidak ada crowding informasi

### 2. **Progressive Analysis**
- Melihat perkembangan split dari atas ke bawah
- Memahami bagaimana data training bertambah
- Analisis temporal yang jelas

### 3. **Enhanced Readability**
- Ukuran gambar yang lebih besar
- Informasi yang lebih detail
- Layout yang lebih terorganisir

### 4. **Comprehensive Statistics**
- Informasi periode waktu yang jelas
- Jumlah samples yang akurat
- Format tanggal yang mudah dibaca

## 📋 Cara Menjalankan

### Test Script:
```bash
python test_timeseries_split.py
```

### Full Training:
```bash
python final_main_gbm.py
```

## 🔍 Interpretasi Visualisasi

### Membaca Plot:
1. **Overall Title**: Nama target dan informasi umum
2. **Split Progression**: Dari Split 1 (atas) ke Split 5 (bawah)
3. **Timeline**: Sumbu X menunjukkan waktu
4. **Values**: Sumbu Y menunjukkan nilai target
5. **Train/Val**: Garis solid (train) vs putus-putus (validation)
6. **Statistics**: Box informasi di setiap subplot

### Analisis Split:
- **Split 1**: Training dengan data awal, validation dengan data tengah
- **Split 2**: Training dengan data lebih banyak, validation dengan data lebih lanjut
- **Split 3-5**: Progresif menggunakan data training yang semakin banyak

Visualisasi ini memberikan pemahaman yang lebih detail dan terorganisir untuk setiap parameter target, memungkinkan analisis yang lebih mendalam terhadap karakteristik masing-masing variabel cuaca. 