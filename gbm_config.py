import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.inspection import permutation_importance
import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import clone
import warnings
from tqdm import tqdm  # For progress bars
import time

# Configure parallel processing
import multiprocessing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)  # Use half of available CPUs for better stability

# Configure numpy for better performance
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# Suppress warnings
warnings.filterwarnings('ignore')

# Create base directory for all GBM results
GBM_DIR = 'gbm'
MODELS_DIR = os.path.join(GBM_DIR, 'models')
PLOTS_DIR = os.path.join(GBM_DIR, 'plots')
LOGS_DIR = os.path.join(GBM_DIR, 'logs')

# Create necessary directories
os.makedirs(GBM_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Target variables configuration
TARGET_COLUMNS = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
TARGET_NAMES = {
    'RR': 'Rainfall (mm)',
    'ss': 'Sunshine Duration (hours)',
    'Tavg': 'Average Temperature (°C)',
    'ddd_car': 'Wind Direction',
    'ff_avg': 'Wind Speed (m/s)'
}

# Feature columns configuration
FEATURE_COLUMNS = [
    # Basic weather measurements (excluding targets)
    'Tn', 'Tx', 'RH_avg',    # Temperature min/max, humidity
    
    # Essential temporal features (to capture seasonality)
    'Month_sin', 'Month_cos',    # Cyclical month encoding
    
    # Key derived features
    'Temp_Range',    # Temperature range (important for weather prediction)
    'Dew_Point',     # Dew point (crucial for various weather variables)
    
    # Recent history features (most recent are most important)
    'RR_Rolling_Mean_3d',    # 3-day rainfall moving average
    'RR_Rolling_Std_3d',     # 3-day rainfall standard deviation
    'RH_Rolling_Mean_3d',    # 3-day humidity moving average
    
    # Recent lag features (most recent lags are most important)
    'RR_Lag_1',             # Previous day's rainfall
    'RR_Lag_2',             # 2 days ago rainfall
    'Rain_Binary_Lag_1',    # Whether it rained yesterday
    
    # Rain pattern indicators
    'Rain_Streak',    # Consecutive days with rain
    'Dry_Streak'      # Consecutive days without rain
]

# Model hyperparameters for different target types
MODEL_CONFIGS = {
    'ddd_car': {  # Wind direction - circular regression
        'n_estimators': 150,
        'learning_rate': 0.08,
        'max_depth': 4,
        'min_samples_split': 8,
        'min_samples_leaf': 6,
        'subsample': 0.8,
        'random_state': 42
    },
    'RR': {  # Rainfall - can be zero-inflated
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'subsample': 0.8,
        'random_state': 42
    },
    'ss': {  # Sunshine - can be zero-inflated
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'subsample': 0.8,
        'random_state': 42
    },
    'default': {  # Temperature and wind speed - more continuous
        'n_estimators': 180,
        'learning_rate': 0.06,
        'max_depth': 4,
        'min_samples_split': 6,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'random_state': 42
    }
}

# Forecasting configuration
FORECAST_DAYS = 5  # Number of days to forecast ahead
CV_FOLDS = 5  # Number of time series splits

# Target constraints for prediction validation
TARGET_CONSTRAINTS = {
    'RR': {'min': 0, 'max': None},  # Rainfall cannot be negative
    'ss': {'min': 0, 'max': 24},    # Sunshine hours: 0-24 hours
    'ff_avg': {'min': 0, 'max': None},  # Wind speed cannot be negative
    'ddd_car': {'min': 0, 'max': 360}   # Wind direction: 0-360 degrees
}

# Scaling configuration for targets
TARGETS_REQUIRING_SCALING = ['RR', 'ss', 'ff_avg']  # These benefit from scaling 