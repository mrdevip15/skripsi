"""
GBM Weather Predictor Package

A comprehensive weather prediction system using Gradient Boosting Machines
for multi-target weather forecasting including rainfall, sunshine, temperature,
wind direction, and wind speed.
"""

__version__ = "1.0.0"
__author__ = "Weather Prediction Team"

from .models.gbm_predictor import GBMWeatherPredictor
from .data.preprocessing import DataPreprocessor
from .data.feature_engineering import FeatureEngineer
from .visualization.plotting import WeatherPlotter

__all__ = [
    'GBMWeatherPredictor',
    'DataPreprocessor', 
    'FeatureEngineer',
    'WeatherPlotter'
] 