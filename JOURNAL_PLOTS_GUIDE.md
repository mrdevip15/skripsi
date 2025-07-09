# Journal Plot Generation Guide

## 🎨 Overview

The GBM Weather Predictor now automatically generates **publication-quality journal plots** with vibrant colors and professional styling. These plots are designed specifically for academic publications and research papers.

## 📊 Generated Plots

### **1. Figure 1: Target Variable Distributions**
- **File**: `Figure_1_Target_Distributions.png`
- **Content**: Histograms with KDE for all target variables (RR, ss, Tavg, ddd_car, ff_avg)
- **Features**: 
  - Color-coded distributions
  - Statistical annotations (mean, standard deviation)
  - Professional styling with grid lines
  - High-resolution (300 DPI)

### **2. Figure 2: Feature Correlation Matrix**
- **File**: `Figure_2_Feature_Correlation_Matrix.png`
- **Content**: Heatmap showing correlations between features and target variables
- **Features**:
  - Diverging color palette (blue to red)
  - Annotated correlation values
  - Top 15 most correlated features
  - Professional heatmap styling

### **3. Figure 3: Multi-Target Feature Importance**
- **File**: `Figure_3_Multi_Target_Feature_Importance.png`
- **Content**: Horizontal bar charts showing top 8 features for each target
- **Features**:
  - Color-coded by target variable
  - Value labels on bars
  - Grid lines for readability
  - Consistent color scheme

### **4. Figure 4: Seasonal Patterns**
- **File**: `Figure_4_Seasonal_Patterns.png`
- **Content**: Monthly averages for all target variables
- **Features**:
  - Bar charts with value labels
  - Indonesian month names
  - Color-coded by variable
  - Grid lines for reference

### **5. Figure 5: Prediction Performance**
- **File**: `Figure_5_Prediction_Performance.png`
- **Content**: Performance metrics (R², RMSE, MAE) for all targets
- **Features**:
  - Side-by-side bar charts
  - Value labels on bars
  - Appropriate y-axis scaling
  - Color-coded metrics

### **6. Figure 6: Actual vs Predicted Plots**
- **Files**: `Figure_6_*.png` (one for each target)
- **Content**: Time series and scatter plots for each target
- **Features**:
  - Dual-panel layout (time series + scatter)
  - Perfect prediction line
  - R² value annotation
  - Professional styling

## 🎨 Color Scheme

The journal plots use a carefully selected color palette:

```python
colors = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple  
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'info': '#3A1772',         # Dark Purple
    'warning': '#F4A261',      # Light Orange
    'light': '#E9C46A',        # Yellow
    'dark': '#264653'          # Dark Blue
}
```

## 📐 Publication Standards

### **Resolution & Quality**
- **DPI**: 300 (print quality)
- **Format**: PNG (lossless)
- **Size**: Optimized for journal layouts
- **Font**: Professional typography

### **Styling Features**
- **Grid lines**: Subtle alpha for readability
- **Annotations**: Statistical values displayed
- **Legends**: Clear and positioned appropriately
- **Titles**: Bold, descriptive titles
- **Labels**: Proper axis labels and tick marks

## 🚀 How to Use

### **Automatic Generation**
The journal plots are generated automatically when you run the main script:

```bash
python3 run_organized_gbm.py
```

### **Manual Generation**
You can also generate journal plots manually:

```python
from gbm_weather_predictor.visualization import JournalPlotter

# Create journal plotter
plotter = JournalPlotter("my_journal_plots")

# Generate all plots
plotter.create_comprehensive_journal_plots(df, target_models, feature_columns)
```

### **Individual Plot Generation**
Generate specific plots:

```python
# Target distributions
plotter.create_target_distribution_plot(df)

# Correlation heatmap
plotter.create_correlation_heatmap(df, feature_columns)

# Feature importance
plotter.create_feature_importance_plot(target_models, feature_columns)

# Seasonal patterns
plotter.create_seasonal_patterns_plot(df)

# Performance comparison
plotter.create_prediction_performance_plot(target_models)

# Actual vs predicted
plotter.create_actual_vs_predicted_plot(dates, actual, predicted, target_name)
```

## 📁 Output Structure

```
journal_plots/
├── Figure_1_Target_Distributions.png
├── Figure_2_Feature_Correlation_Matrix.png
├── Figure_3_Multi_Target_Feature_Importance.png
├── Figure_4_Seasonal_Patterns.png
├── Figure_5_Prediction_Performance.png
├── Figure_6_curah_hujan_mm_Predictions.png
├── Figure_6_lama_penyinaran_matahari_jam_Predictions.png
├── Figure_6_suhu_rata_rata_c_Predictions.png
├── Figure_6_arah_angin_Predictions.png
└── Figure_6_kecepatan_angin_ms_Predictions.png
```

## 🎯 Publication Ready Features

### **1. Academic Standards**
- ✅ High resolution (300 DPI)
- ✅ Professional color scheme
- ✅ Clear typography
- ✅ Proper annotations
- ✅ Statistical information

### **2. Journal Compatibility**
- ✅ Standard figure sizes
- ✅ CMYK color space ready
- ✅ Vector graphics support
- ✅ Multiple format options
- ✅ Accessibility considerations

### **3. Research Quality**
- ✅ Comprehensive analysis
- ✅ Multiple visualization types
- ✅ Statistical rigor
- ✅ Clear interpretation
- ✅ Professional presentation

## 🔧 Customization

### **Color Scheme**
Modify colors in `journal_plotter.py`:

```python
self.colors = {
    'primary': '#YOUR_COLOR',
    'secondary': '#YOUR_COLOR',
    # ... more colors
}
```

### **Figure Sizes**
Adjust figure dimensions:

```python
plt.figure(figsize=(18, 12))  # Width, Height in inches
```

### **Font Sizes**
Modify typography:

```python
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    # ... more settings
})
```

## 📈 Usage Examples

### **For Research Papers**
1. Run the complete pipeline
2. Use generated plots in your manuscript
3. Include statistical annotations
4. Reference figure numbers

### **For Presentations**
1. Generate plots with larger fonts
2. Use for conference presentations
3. Include in slides
4. Export for posters

### **For Reports**
1. Generate comprehensive analysis
2. Include in technical reports
3. Use for stakeholder presentations
4. Export for documentation

## 🎉 Benefits

### **1. Time Saving**
- ✅ Automatic generation
- ✅ No manual plotting
- ✅ Consistent styling
- ✅ Batch processing

### **2. Quality Assurance**
- ✅ Publication standards
- ✅ Professional appearance
- ✅ Statistical accuracy
- ✅ Visual clarity

### **3. Research Impact**
- ✅ Journal-ready figures
- ✅ Clear communication
- ✅ Professional presentation
- ✅ Academic credibility

## 🚀 Quick Start

1. **Run the pipeline**:
   ```bash
   python3 run_organized_gbm.py
   ```

2. **Check generated plots**:
   ```bash
   ls journal_plots/
   ```

3. **Use in your publication**:
   - Copy plots to your manuscript
   - Reference figure numbers
   - Include in appendices

## 📊 Example Output

The journal plots will show:
- **Vibrant, professional colors**
- **Clear statistical information**
- **High-resolution graphics**
- **Publication-ready formatting**
- **Comprehensive analysis**

Perfect for academic publications, research papers, and professional presentations! 