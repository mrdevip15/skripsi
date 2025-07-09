# GBM Weather Predictor - Organized Structure

This is the organized version of the GBM Weather Predictor, split into logical modules for better maintainability and organization.

## 📁 Project Structure

```
gbm_weather_predictor/
├── __init__.py                 # Main package initialization
├── main.py                     # Main entry point
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration constants and settings
├── data/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing functionality
│   └── feature_engineering.py  # Feature engineering functionality
├── models/
│   ├── __init__.py
│   └── gbm_predictor.py       # Main GBM predictor class
├── visualization/
│   ├── __init__.py
│   └── plotting.py            # Visualization and plotting functions
└── utils/
    ├── __init__.py
    └── helpers.py             # Utility helper functions
```

## 🚀 Quick Start

### Option 1: Using the runner script
```bash
python run_organized_gbm.py
```

### Option 2: Using the module directly
```python
from gbm_weather_predictor.main import main
main()
```

## 📋 Module Descriptions

### **config/settings.py**
- Configuration constants (TARGET_COLUMNS, TARGET_NAMES, etc.)
- Model hyperparameters for different targets
- Directory paths and environment setup
- Wind direction mapping

### **data/preprocessing.py**
- `DataPreprocessor` class
- Handles special values (8888, 9999)
- Wind direction conversion
- Missing value interpolation
- Data cleaning and validation

### **data/feature_engineering.py**
- `FeatureEngineer` class
- Temporal feature creation (Month, Day, DayOfWeek, etc.)
- Cyclical encoding (sin/cos transformations)
- Rolling statistics (mean, std)
- Lag features
- Weather physics-based features (dew point, heat index)
- Rain streak calculations

### **models/gbm_predictor.py**
- `GBMWeatherPredictor` class
- Multi-target model training
- Cross-validation
- Model evaluation and metrics
- Model saving/loading
- Prediction constraints application

### **visualization/plotting.py**
- `WeatherPlotter` class
- Target distribution plots
- Feature correlation heatmaps
- Feature importance plots
- Seasonal pattern plots
- Prediction vs actual plots

### **utils/helpers.py**
- Utility functions for file operations
- Metrics saving/loading
- Timestamp generation
- Filename sanitization
- Directory management

## 🔧 Key Features

1. **Modular Design**: Each component is isolated and can be tested independently
2. **Configuration Management**: All settings centralized in config/settings.py
3. **Data Pipeline**: Clear separation between preprocessing and feature engineering
4. **Visualization**: Dedicated plotting module for all visualizations
5. **Error Handling**: Comprehensive error handling throughout the pipeline
6. **Logging**: Detailed logging for debugging and monitoring

## 📊 Output Structure

The organized version produces the same outputs as the original:

```
gbm/
├── models/
│   ├── target_RR/
│   ├── target_ss/
│   ├── target_Tavg/
│   ├── target_ddd_car/
│   ├── target_ff_avg/
│   ├── feature_scaler.pkl
│   └── feature_columns.pkl
├── plots/
│   └── YYYYMMDD_HHMMSS/
│       ├── feature_correlations.png
│       ├── feature_distributions.png
│       ├── seasonal_patterns.png
│       ├── multi_target_feature_importance.png
│       ├── predictions_*.png
│       └── metrics_*.txt
└── logs/
```

## 🎯 Benefits of Organization

1. **Maintainability**: Easy to locate and modify specific functionality
2. **Reusability**: Components can be imported and used independently
3. **Testing**: Each module can be unit tested separately
4. **Documentation**: Clear structure makes it easier to understand
5. **Collaboration**: Multiple developers can work on different modules
6. **Scalability**: Easy to add new features or modify existing ones

## 🔄 Migration from Original

The organized version maintains full compatibility with the original functionality while providing better structure. All the original features are preserved:

- Multi-target weather prediction
- Advanced feature engineering
- Comprehensive visualization
- Model persistence
- Performance metrics

## 📝 Usage Examples

### Basic Usage
```python
from gbm_weather_predictor import GBMWeatherPredictor
from gbm_weather_predictor.data import DataPreprocessor, FeatureEngineer

# Load and preprocess data
preprocessor = DataPreprocessor()
processed_df = preprocessor.preprocess_data(df)

# Engineer features
feature_engineer = FeatureEngineer()
processed_df = feature_engineer.engineer_features(processed_df)

# Train models
predictor = GBMWeatherPredictor()
metrics = predictor.train_and_evaluate(processed_df)
```

### Custom Configuration
```python
from gbm_weather_predictor.config.settings import MODEL_CONFIGS

# Modify model configuration
MODEL_CONFIGS['RR']['n_estimators'] = 300
MODEL_CONFIGS['RR']['learning_rate'] = 0.03
```

### Visualization Only
```python
from gbm_weather_predictor.visualization import WeatherPlotter

plotter = WeatherPlotter('output_dir')
plotter.plot_feature_correlations(df, feature_columns)
plotter.plot_seasonal_patterns(df)
``` 