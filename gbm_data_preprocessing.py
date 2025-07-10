import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from gbm_config import *

class DataPreprocessor:
    """Handles all data preprocessing tasks including cleaning, feature engineering, and preparation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scalers = {}  # Separate scalers for each target
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with interpolation for special values"""
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            print(f"Initial dataset size: {initial_size}")
            
            # Convert date
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            
            # Handle special values (8888 and 9999) with enhanced interpolation
            special_values = [8888, 9999]
            
            # Identify all columns with special values
            columns_to_interpolate = []
            for col in df.columns:
                if df[col].dtype in [np.int64, np.float64]:
                    if any(df[col].isin(special_values)):
                        columns_to_interpolate.append(col)
            
            if columns_to_interpolate:
                print(f"\nFound special values (8888, 9999) in columns: {', '.join(columns_to_interpolate)}")
                
                # Sort by date for proper interpolation
                df = df.sort_values('Tanggal')
                
                # Process each column with special values using enhanced interpolation
                for col in columns_to_interpolate:
                    # Count special values before replacement
                    special_count = sum(df[col].isin(special_values))
                    
                    if special_count > 0:
                        print(f"Column '{col}': {special_count} special values found")
                        
                        # Create a mask for special values
                        mask = df[col].isin(special_values)
                        
                        # Store original values for reference
                        original_special_positions = df.index[mask].tolist()
                        
                        # Temporarily set special values to NaN for interpolation
                        df.loc[mask, col] = np.nan
                        
                        # Step 1: Forward fill (propagate last valid observation forward)
                        df[col] = df[col].fillna(method='ffill')
                        filled_by_ffill = df.loc[original_special_positions, col].notna().sum()
                        
                        # Step 2: Backward fill (propagate next valid observation backward)
                        df[col] = df[col].fillna(method='bfill')
                        filled_by_bfill = df.loc[original_special_positions, col].notna().sum() - filled_by_ffill
                        
                        # Step 3: For any remaining NaN values, use 5-day rolling mean
                        remaining_nan_mask = df[col].isna()
                        if remaining_nan_mask.any():
                            # Calculate 5-day rolling mean (centered window when possible)
                            rolling_mean_5d = df[col].rolling(window=5, center=True, min_periods=1).mean()
                            
                            # Fill remaining NaN values with 5-day rolling mean
                            df.loc[remaining_nan_mask, col] = rolling_mean_5d.loc[remaining_nan_mask]
                            filled_by_rolling = remaining_nan_mask.sum()
                        else:
                            filled_by_rolling = 0
                        
                        # Final fallback: if any NaNs still remain, use column median
                        final_nan_mask = df[col].isna()
                        if final_nan_mask.any():
                            column_median = df[col].median()
                            df.loc[final_nan_mask, col] = column_median
                            filled_by_median = final_nan_mask.sum()
                            print(f"  - Used median ({column_median:.2f}) for {filled_by_median} remaining values")
                        
                        print(f"  - Filled {filled_by_ffill} values with forward fill")
                        print(f"  - Filled {filled_by_bfill} values with backward fill")
                        print(f"  - Filled {filled_by_rolling} values with rolling mean")
                        if final_nan_mask.any():
                            print(f"  - Filled {filled_by_median} values with median")
            
            # Create temporal features
            print("\nCreating temporal features...")
            df = self._create_temporal_features(df)
            
            # Create derived features
            print("Creating derived features...")
            df = self._create_derived_features(df)
            
            # Create lag features
            print("Creating lag features...")
            df = self._create_lag_features(df)
            
            # Create rolling statistics
            print("Creating rolling statistics...")
            df = self._create_rolling_features(df)
            
            # Create rain pattern features
            print("Creating rain pattern features...")
            df = self._create_rain_pattern_features(df)
            
            # Remove rows with any remaining NaN values
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)
            removed_rows = initial_rows - final_rows
            
            if removed_rows > 0:
                print(f"\nRemoved {removed_rows} rows with missing values")
                print(f"Final dataset size: {final_rows}")
            
            # Ensure all numeric columns are float
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"\nPreprocessing completed. Final dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _create_temporal_features(self, df):
        """Create temporal features from date"""
        df = df.copy()
        
        # Extract date components
        df['Year'] = df['Tanggal'].dt.year
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        
        # Create cyclical encoding for month (to handle seasonality)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Create cyclical encoding for day of year
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
        
        return df
    
    def _create_derived_features(self, df):
        """Create derived features from existing variables"""
        df = df.copy()
        
        # Temperature range
        if 'Tn' in df.columns and 'Tx' in df.columns:
            df['Temp_Range'] = df['Tx'] - df['Tn']
        
        # Dew point calculation (approximate)
        if 'Tavg' in df.columns and 'RH_avg' in df.columns:
            # Magnus formula for dew point
            a = 17.27
            b = 237.7
            alpha = ((a * df['Tavg']) / (b + df['Tavg'])) + np.log(df['RH_avg'] / 100)
            df['Dew_Point'] = (b * alpha) / (a - alpha)
        
        # Heat index (approximate)
        if 'Tavg' in df.columns and 'RH_avg' in df.columns:
            # Simplified heat index calculation
            df['Heat_Index'] = df['Tavg'] + 0.5 * (df['RH_avg'] / 100) * (df['Tavg'] - 20)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lag features for time series prediction"""
        df = df.copy()
        
        # Sort by date to ensure proper lag creation
        df = df.sort_values('Tanggal')
        
        # Create lag features for rainfall
        if 'RR' in df.columns:
            for lag in [1, 2, 3, 7]:
                df[f'RR_Lag_{lag}'] = df['RR'].shift(lag)
            
            # Binary rain indicator lags
            df['Rain_Binary_Lag_1'] = (df['RR'].shift(1) > 0).astype(int)
            df['Rain_Binary_Lag_2'] = (df['RR'].shift(2) > 0).astype(int)
            df['Rain_Binary_Lag_3'] = (df['RR'].shift(3) > 0).astype(int)
        
        # Create lag features for other variables
        for col in ['Tavg', 'RH_avg', 'ss', 'ff_avg']:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df):
        """Create rolling window statistics"""
        df = df.copy()
        
        # Sort by date
        df = df.sort_values('Tanggal')
        
        # Rolling statistics for rainfall
        if 'RR' in df.columns:
            for window in [3, 7, 14]:
                df[f'RR_Rolling_Mean_{window}d'] = df['RR'].rolling(window=window, min_periods=1).mean()
                df[f'RR_Rolling_Std_{window}d'] = df['RR'].rolling(window=window, min_periods=1).std()
                df[f'RR_Rolling_Max_{window}d'] = df['RR'].rolling(window=window, min_periods=1).max()
        
        # Rolling statistics for temperature
        if 'Tavg' in df.columns:
            for window in [3, 7]:
                df[f'Tavg_Rolling_Mean_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
                df[f'Tavg_Rolling_Std_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).std()
        
        # Rolling statistics for humidity
        if 'RH_avg' in df.columns:
            for window in [3, 7]:
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
                df[f'RH_Rolling_Std_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).std()
        
        # Rolling statistics for sunshine
        if 'ss' in df.columns:
            for window in [3, 7]:
                df[f'ss_Rolling_Mean_{window}d'] = df['ss'].rolling(window=window, min_periods=1).mean()
                df[f'ss_Rolling_Std_{window}d'] = df['ss'].rolling(window=window, min_periods=1).std()
        
        return df
    
    def _create_rain_pattern_features(self, df):
        """Create rain pattern features"""
        df = df.copy()
        
        if 'RR' in df.columns:
            # Rain streak (consecutive days with rain)
            rain_binary = (df['RR'] > 0).astype(int)
            rain_groups = (rain_binary != rain_binary.shift()).cumsum()
            df['Rain_Streak'] = rain_binary.groupby(rain_groups).cumsum() * rain_binary
            
            # Dry streak (consecutive days without rain)
            dry_binary = (df['RR'] == 0).astype(int)
            dry_groups = (dry_binary != dry_binary.shift()).cumsum()
            df['Dry_Streak'] = dry_binary.groupby(dry_groups).cumsum() * dry_binary
            
            # Rain intensity categories
            df['Rain_Intensity'] = pd.cut(df['RR'], 
                                        bins=[-np.inf, 0, 2.5, 7.5, 50, np.inf],
                                        labels=['No Rain', 'Light', 'Moderate', 'Heavy', 'Very Heavy'])
            
            # Convert to numeric for modeling
            df['Rain_Intensity_Code'] = df['Rain_Intensity'].cat.codes
        
        return df
    
    def prepare_multi_day_dataset(self, df):
        """Prepare dataset for multi-day forecasting"""
        try:
            print("Preparing multi-day forecasting dataset...")
            
            # Ensure we have all required features
            available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                print(f"Using available features: {available_features}")
            
            # Create multi-day targets
            multi_day_data = {}
            
            for target in TARGET_COLUMNS:
                if target not in df.columns:
                    print(f"Skipping {target} - not found in dataset")
                    continue
                
                print(f"Preparing {target} for multi-day forecasting...")
                
                # Create future targets (next 1-5 days)
                target_data = {}
                for day in range(1, FORECAST_DAYS + 1):
                    target_data[f'{target}_Day_{day}'] = df[target].shift(-day)
                
                # Combine features and targets
                feature_data = df[available_features].copy()
                target_df = pd.DataFrame(target_data)
                
                # Combine features and targets
                combined_data = pd.concat([feature_data, target_df], axis=1)
                
                # Remove rows with NaN values (at the end of the dataset)
                combined_data = combined_data.dropna()
                
                multi_day_data[target] = combined_data
                
                print(f"  - {target}: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")
            
            return multi_day_data
            
        except Exception as e:
            print(f"Error preparing multi-day dataset: {str(e)}")
            raise 