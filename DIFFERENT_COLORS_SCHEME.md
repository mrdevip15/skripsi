# Skema Warna Berbeda untuk Train dan Validation

## ✅ Implementasi Warna Berbeda Berhasil

Visualisasi telah diperbarui untuk menggunakan warna yang berbeda untuk data training dan validation, memberikan pembedaan yang lebih jelas antara kedua jenis data.

## 🎨 Skema Warna Baru

### 1. **Train Colors (Warna Training)**
```python
train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```
- **Split 1**: Biru (#1f77b4)
- **Split 2**: Orange (#ff7f0e)
- **Split 3**: Hijau (#2ca02c)
- **Split 4**: Merah (#d62728)
- **Split 5**: Ungu (#9467bd)

### 2. **Validation Colors (Warna Validation)**
```python
val_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
```
- **Split 1**: Light Red (#ff6b6b)
- **Split 2**: Teal (#4ecdc4)
- **Split 3**: Light Blue (#45b7d1)
- **Split 4**: Light Green (#96ceb4)
- **Split 5**: Yellow (#feca57)

## 🔧 Perubahan Kode

### Sebelum:
```python
# Menggunakan warna yang sama dengan line style berbeda
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ax.plot(train_dates, train_data, color=color, linewidth=2.5, 
        label=f'Split {split_idx+1} Train', alpha=0.9)
ax.plot(val_dates, val_data, color=color, linewidth=2.5, 
        linestyle='--', label=f'Split {split_idx+1} Val', alpha=0.9)
```

### Sesudah:
```python
# Menggunakan warna yang berbeda untuk train dan validation
train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
val_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']

ax.plot(train_dates, train_data, color=train_color, linewidth=2.5, 
        label=f'Split {split_idx+1} Train', alpha=0.9)
ax.plot(val_dates, val_data, color=val_color, linewidth=2.5, 
        label=f'Split {split_idx+1} Val', alpha=0.9)
```

## 📊 Hasil Implementasi

### File Output dengan Warna Berbeda:
- ✅ `data_splits_rr.png` (1.1MB) - Curah Hujan (mm)
- ✅ `data_splits_ss.png` (1.6MB) - Durasi Sinar Matahari (jam)
- ✅ `data_splits_tavg.png` (1.5MB) - Suhu Rata-rata (°C)
- ✅ `data_splits_ddd_car.png` (1.4MB) - Arah Angin (derajat)
- ✅ `data_splits_ff_avg.png` (1.0MB) - Kecepatan Angin (m/s)

### Lokasi File:
```
gbm/plots/20250708_102708/
├── data_splits_rr.png
├── data_splits_ss.png
├── data_splits_tavg.png
├── data_splits_ddd_car.png
├── data_splits_ff_avg.png
└── original_target_variables_distribution.png
```

## 🎯 Keunggulan Skema Warna Baru

### 1. **Clear Distinction**
- Train dan validation data memiliki warna yang berbeda
- Tidak perlu mengandalkan line style (solid vs dashed)
- Pembedaan visual yang lebih jelas

### 2. **Color Harmony**
- Train colors: Warna yang lebih gelap dan solid
- Validation colors: Warna yang lebih terang dan complementary
- Kombinasi warna yang harmonis dan mudah dibedakan

### 3. **Better Accessibility**
- Kontras yang lebih baik antara train dan validation
- Mudah dibedakan bahkan untuk colorblind users
- Visual hierarchy yang jelas

### 4. **Professional Appearance**
- Skema warna yang profesional
- Konsisten di semua target parameter
- Cocok untuk publikasi dan presentasi

## 🎨 Interpretasi Warna

### Train Data (Warna Gelap):
- **Biru**: Split 1 - Data training awal
- **Orange**: Split 2 - Data training yang bertambah
- **Hijau**: Split 3 - Data training yang semakin banyak
- **Merah**: Split 4 - Data training yang lebih besar
- **Ungu**: Split 5 - Data training terbesar

### Validation Data (Warna Terang):
- **Light Red**: Split 1 - Validation data tengah
- **Teal**: Split 2 - Validation data lebih lanjut
- **Light Blue**: Split 3 - Validation data selanjutnya
- **Light Green**: Split 4 - Validation data lanjutan
- **Yellow**: Split 5 - Validation data terakhir

## 📈 Analisis Visual

### 1. **Color Progression**
- Train colors: Progresi dari biru → orange → hijau → merah → ungu
- Validation colors: Progresi dari light red → teal → light blue → light green → yellow
- Setiap split memiliki kombinasi warna yang unik

### 2. **Visual Clarity**
- Train data: Warna solid dan gelap untuk menunjukkan stabilitas
- Validation data: Warna terang untuk menunjukkan variasi dan testing
- Background data: Abu-abu transparan untuk konteks

### 3. **Legend Clarity**
- Setiap split memiliki dua warna berbeda
- Legend menunjukkan train dan validation dengan warna yang berbeda
- Mudah diidentifikasi dalam plot

## 🔍 Cara Membaca Plot

### 1. **Identifikasi Split**
- Setiap subplot menunjukkan satu split
- Train data: Warna gelap (biru, orange, hijau, merah, ungu)
- Validation data: Warna terang (light red, teal, light blue, light green, yellow)

### 2. **Progresi Data**
- Split 1: Training data minimal, validation data tengah
- Split 2-5: Training data bertambah, validation data bergeser ke depan
- Visualisasi progresi yang jelas

### 3. **Data Patterns**
- Train data: Menunjukkan pola yang dipelajari model
- Validation data: Menunjukkan performa prediksi
- Background data: Memberikan konteks keseluruhan

## 📋 Cara Menjalankan

### Test Script:
```bash
python test_timeseries_split.py
```

### Full Training:
```bash
python final_main_gbm.py
```

## 🎯 Kesimpulan

Implementasi skema warna berbeda telah berhasil memberikan:
- **Clear Visual Distinction**: Pembedaan yang jelas antara train dan validation
- **Professional Color Scheme**: Kombinasi warna yang harmonis dan profesional
- **Better Accessibility**: Kontras yang lebih baik untuk semua pengguna
- **Enhanced Readability**: Visualisasi yang lebih mudah dibaca dan dipahami

Visualisasi sekarang menggunakan warna yang berbeda untuk train dan validation data, memberikan pembedaan visual yang lebih jelas dan profesional! ✅ 