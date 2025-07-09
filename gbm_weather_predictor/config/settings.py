"""
Configuration settings for GBM Weather Predictor
"""

import os
import multiprocessing

# Configure parallel processing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)

# Configure numpy for better performance
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

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
    'RR': 'Curah Hujan (mm)',
    'ss': 'Lama Penyinaran Matahari (jam)',
    'Tavg': 'Suhu Rata-rata (°C)',
    'ddd_car': 'Arah Angin',
    'ff_avg': 'Kecepatan Angin (m/s)'
}

# Wind direction mapping
WIND_DIR_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

# Model configurations for different targets
MODEL_CONFIGS = {
    'RR': {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5,
        'min_samples_split': 5, 'min_samples_leaf': 4, 'subsample': 0.8,
        'max_features': 'sqrt', 'random_state': 42
    },
    'ss': {
        'n_estimators': 180, 'learning_rate': 0.06, 'max_depth': 4,
        'min_samples_split': 6, 'min_samples_leaf': 5, 'subsample': 0.8,
        'max_features': 'sqrt', 'random_state': 42
    },
    'Tavg': {
        'n_estimators': 150, 'learning_rate': 0.08, 'max_depth': 4,
        'min_samples_split': 8, 'min_samples_leaf': 6, 'subsample': 0.8,
        'max_features': 'sqrt', 'random_state': 42
    },
    'ddd_car': {
        'n_estimators': 150, 'learning_rate': 0.08, 'max_depth': 4,
        'min_samples_split': 8, 'min_samples_leaf': 6, 'subsample': 0.8,
        'max_features': 'sqrt', 'random_state': 42
    },
    'ff_avg': {
        'n_estimators': 160, 'learning_rate': 0.07, 'max_depth': 4,
        'min_samples_split': 7, 'min_samples_leaf': 5, 'subsample': 0.8,
        'max_features': 'sqrt', 'random_state': 42
    }
} 