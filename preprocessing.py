import pandas as pd
import numpy as np
from scipy import interpolate
import logging

def convert_wind_direction(direction):
    """Convert wind direction text to degrees with interpolation for missing values"""
    direction_dict = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    
    if pd.isna(direction):
        return np.nan
    
    direction = str(direction).strip().upper()
    return direction_dict.get(direction, np.nan)

def preprocess_data(df, config):
    """Preprocess weather data with improved handling of missing values"""
    df = df.copy()
    
    logging.info(f"Initial data shape: {df.shape}")
    initial_rows = len(df)
    
    try:
        # Convert date column
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        
        # Convert wind directions and interpolate missing values
        df['ddd_car'] = df['ddd_car'].str.strip()
        df['ddd_car'] = df['ddd_car'].apply(convert_wind_direction)
        df['ddd_car'] = df['ddd_car'].interpolate(method='linear')
        
        # Convert numeric columns
        for column in df.columns:
            if column != 'Tanggal':
                df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Remove invalid rainfall values
        df = df[~df['RR'].isin([8888, 9999])]
        
        # Remove rows where RR is greater than 60 mm
        df = df[df['RR'] <= 60]
        
        rows_after_invalid = len(df)
        
        # Interpolate missing values instead of dropping
        df = df.set_index('Tanggal')
        df = df.interpolate(method='time')
        df = df.reset_index()
        
        # Normalize features
        numerical_cols = [col for col in df.columns if col not in ['Tanggal', 'RR']]
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
        
        # Log results
        logging.info("\nPreprocessing Results:")
        logging.info(f"Initial rows: {initial_rows}")
        logging.info(f"Rows after removing invalid RR: {rows_after_invalid}")
        logging.info(f"Final rows: {len(df)}")
        
        logging.info("\nRainfall Statistics:")
        logging.info(f"Mean: {df['RR'].mean():.2f}")
        logging.info(f"Std: {df['RR'].std():.2f}")
        logging.info(f"Min: {df['RR'].min():.2f}")
        logging.info(f"Max: {df['RR'].max():.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def engineer_features(df, config):
    """Engineer features with improved efficiency"""
    if not config['USE_FEATURE_ENGINEERING']:
        return df
    
    df = df.copy()
    logging.info(f"Initial shape: {df.shape}")
    
    # Add time-based features
    df['day_of_year'] = df['Tanggal'].dt.dayofyear
    df['month'] = df['Tanggal'].dt.month
    df['day_of_week'] = df['Tanggal'].dt.dayofweek
    
    # Vectorized rolling statistics
    for window in config['ROLLING_WINDOW_SIZES']:
        roll = df[config['INPUT_FEATURES']].rolling(window=window)
        
        # Calculate statistics in one go
        means = roll.mean().add_suffix(f'_rolling_mean_{window}d')
        stds = roll.std().add_suffix(f'_rolling_std_{window}d')
        
        df = pd.concat([df, means, stds], axis=1)
    
    # Forward fill NaN values from rolling calculations
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    logging.info(f"Final shape: {df.shape}")
    return df