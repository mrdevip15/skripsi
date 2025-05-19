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
    """Preprocess weather data"""
    df = df.copy()
    
    # Print initial data head and shape
    logging.info("\nInitial data head:")
    logging.info(f"\n{df.head().to_string()}")
    logging.info(f"Initial data shape: {df.shape}")
    initial_rows = len(df)
    
    try:
        # Keep only required columns
        required_columns = ['DATE'] + config['INPUT_FEATURES']
        df = df[required_columns]
        
        # Print data after selecting columns
        logging.info("\nData after selecting required columns:")
        logging.info(f"\n{df.head().to_string()}")
        
        # Convert date column
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Handle 'T' values (trace amounts) and convert to numeric
        for column in df.columns:
            if column != 'DATE':
                df[column] = df[column].replace('T', '0')
                df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Print data after numeric conversion
        logging.info("\nData after numeric conversion:")
        logging.info(f"\n{df.head().to_string()}")
        
        # Interpolate missing values instead of dropping
        df = df.set_index('DATE')
        df = df.interpolate(method='time')
        df = df.reset_index()
        
        # Normalize features (except PRCP which is our target)
        numerical_cols = [col for col in df.columns if col not in ['DATE', 'PRCP']]
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
        
        # Print final preprocessed data
        logging.info("\nFinal preprocessed data:")
        logging.info(f"\n{df.head().to_string()}")
        
        # Log results
        logging.info("\nPreprocessing Results:")
        logging.info(f"Initial rows: {initial_rows}")
        logging.info(f"Final rows: {len(df)}")
        
        for col in numerical_cols + ['PRCP']:
            logging.info(f"\n{col} Statistics:")
            logging.info(f"Mean: {df[col].mean():.2f}")
            logging.info(f"Std: {df[col].std():.2f}")
            logging.info(f"Min: {df[col].min():.2f}")
            logging.info(f"Max: {df[col].max():.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def engineer_features(df, config):
    """Engineer features with improved efficiency"""
    if not config['USE_FEATURE_ENGINEERING']:
        return df
    
    df = df.copy()
    logging.info("\nData before feature engineering:")
    logging.info(f"\n{df.head().to_string()}")
    logging.info(f"Initial shape: {df.shape}")
    
    # Add time-based features
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # Add interaction features
    df['temp_range'] = df['TMAX'] - df['TMIN']
    df['snow_coverage'] = (df['SNWD'] > 0).astype(int)
    
    # Print data after basic feature engineering
    logging.info("\nData after basic feature engineering:")
    logging.info(f"\n{df.head().to_string()}")
    
    # Add lagged features
    for feature in config['INPUT_FEATURES']:
        for lag in [1, 2, 3]:  # Add 1-day, 2-day, and 3-day lags
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Vectorized rolling statistics with more windows
    for window in config['ROLLING_WINDOW_SIZES']:
        roll = df[config['INPUT_FEATURES']].rolling(window=window)
        
        # Calculate more statistics
        means = roll.mean().add_suffix(f'_rolling_mean_{window}d')
        stds = roll.std().add_suffix(f'_rolling_std_{window}d')
        maxs = roll.max().add_suffix(f'_rolling_max_{window}d')
        mins = roll.min().add_suffix(f'_rolling_min_{window}d')
        
        df = pd.concat([df, means, stds, maxs, mins], axis=1)
    
    # Add exponential moving averages
    for alpha in [0.1, 0.3, 0.5]:
        ema = df[config['INPUT_FEATURES']].ewm(alpha=alpha).mean()
        ema = ema.add_suffix(f'_ema_{int(alpha*10)}')
        df = pd.concat([df, ema], axis=1)
    
    # Forward fill NaN values from rolling calculations
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Print final engineered data
    logging.info("\nFinal engineered data:")
    logging.info(f"\n{df.head().to_string()}")
    logging.info(f"Final shape: {df.shape}")
    
    return df