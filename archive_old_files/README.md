# Weather Prediction with Gradient Boosting

This project uses Gradient Boosting Machine (GBM) regression to predict rainfall based on historical weather data from Makassar, Indonesia.

## Overview

The `main_gbm.py` script trains a Gradient Boosting Regressor model to predict precipitation (RR) based on various weather metrics. It features comprehensive data preprocessing, feature engineering, model training, evaluation, and visualization capabilities.

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Ensure your data file `makassar.csv` is in the same directory as the script
2. Run the script:

```bash
python main_gbm.py
```

3. Results will be saved in the `gbm` directory

## Data Format

The script expects a CSV file with the following columns:
- `Tanggal`: Date (format: DD-MM-YYYY)
- `Tn`: Minimum temperature
- `Tx`: Maximum temperature
- `Tavg`: Average temperature
- `RH_avg`: Average relative humidity
- `RR`: Rainfall (target variable)
- `ss`: Sunshine duration
- `ff_x`: Wind speed
- `ff_avg`: Average wind speed
- `ddd_x`: Wind direction (degrees)
- `ddd_car`: Wind direction (cardinal directions)

## Features

### Data Preprocessing

- Date conversion and feature extraction
- Outlier removal (rainfall > 100mm)
- Missing value imputation
- Wind direction encoding
- Feature engineering (time-based, rolling statistics, lag features)

### Model Training

- Gradient Boosting Regressor with optimized hyperparameters
- Feature standardization
- Chronological train-test split (80/20)
- Comprehensive error handling and validation

### Visualizations

The script generates several visualizations:
- Rainfall distribution (before and after outlier removal)
- Feature correlations heatmap
- Feature distributions with correlation values
- Seasonal patterns of rainfall
- Actual vs. predicted rainfall
- Feature importance

## Project Structure

The project is organized as follows:

```
.
├── main_gbm.py              # Main script for training the GBM model
├── predict.py               # Script for making predictions
├── makassar.csv             # Weather data from Makassar
├── README.md                # This documentation file
├── gbm/                     # Directory for GBM model results
│   ├── models/              # Saved trained models
│   │   └── gbm_model_TIMESTAMP.pkl  # Model with metadata
│   ├── plots/               # Visualization outputs
│   │   └── TIMESTAMP/       # Timestamp-specific run folder
│   │       ├── original_rainfall_distribution.png
│   │       ├── rainfall_distribution_after_outlier_removal.png
│   │       ├── feature_correlations.png
│   │       ├── feature_distributions.png
│   │       ├── seasonal_patterns.png
│   │       ├── predictions.png
│   │       ├── feature_importance.png
│   │       └── metrics.txt  # Performance metrics
│   └── logs/
│       └── gbm_prediction.log  # Detailed logging information
└── archive/                 # Archived code (not used in current implementation)
```

## Model Performance

The model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)

## Customization

You can modify:
- Feature selection: Edit `self.feature_columns` in `__init__`
- Model parameters: Change the parameters in `train_and_evaluate`
- Outlier threshold: Modify the `100` value in the outlier removal section
- Train/test split ratio: Adjust the `0.8` value in `train_and_evaluate`

## Troubleshooting

- **NaN errors**: The script includes extensive checks to handle NaN values. If they persist, check data quality.
- **Memory issues**: For large datasets, reduce the number of features or sampling rate.
- **Low accuracy**: Try adjusting model parameters or adding more engineered features.

## License

[MIT License](LICENSE)
 
