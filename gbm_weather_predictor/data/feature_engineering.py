"""
Feature engineering module for GBM Weather Predictor
"""

import pandas as pd
import numpy as np
from ..config.settings import TARGET_COLUMNS

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
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
        
    def engineer_features(self, df):
        """Add engineered features to the dataset"""
        df = df.copy()
        
        # Add basic features that won't introduce NaN values
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfWeek'] = df['Tanggal'].dt.dayofweek
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        df['Season'] = (df['Month'] % 12 + 3) // 3
        
        # Make sure Tx and Tn are numeric before calculating Temp_Range
        df['Tx'] = pd.to_numeric(df['Tx'], errors='coerce')
        df['Tn'] = pd.to_numeric(df['Tn'], errors='coerce')
        df['Temp_Range'] = df['Tx'] - df['Tn']
        
        # Add cyclical encoding of time features (better for capturing seasonality)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
        
        # Make sure all target variables are numeric before calculations
        for target in TARGET_COLUMNS:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors='coerce')
        
        # Make sure other key variables are numeric
        df['RH_avg'] = pd.to_numeric(df['RH_avg'], errors='coerce')
        
        # Add interaction terms known to be important in meteorology
        # Use Tx and Tn instead of Tavg for interactions since Tavg is now a target
        df['Temp_Humidity'] = ((df['Tx'] + df['Tn']) / 2) * df['RH_avg']  # Using average of Tx and Tn
        df['Temp_Range_RH'] = df['Temp_Range'] * df['RH_avg']
        
        # Adding weather physics-based features
        # Simplified dew point calculation using Tx and Tn average
        temp_avg_calc = (df['Tx'] + df['Tn']) / 2
        df['Dew_Point'] = temp_avg_calc - ((100 - df['RH_avg']) / 5)
        
        # Simplified heat index
        df['Heat_Index'] = temp_avg_calc + 0.05 * df['RH_avg']
        
        # Add rolling features for all target variables with careful handling of NaN values
        windows = [3, 7, 14]  # Adding 2-week window
        for window in windows:
            for target in TARGET_COLUMNS:
                if target in df.columns:
                    # Rolling mean with min_periods=1 to avoid NaN
                    df[f'{target}_Rolling_Mean_{window}d'] = df[target].rolling(window=window, min_periods=1).mean()
                    df[f'{target}_Rolling_Std_{window}d'] = df[target].rolling(window=window, min_periods=1).std()
                
                # Also add rolling features for other important variables
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
        
        # Add lag features for all target variables (more lags with increasing significance)
        for lag in [1, 2, 3, 5, 7, 14]:
            for target in TARGET_COLUMNS:
                if target in df.columns:
                    df[f'{target}_Lag_{lag}'] = df[target].shift(lag)
                    # Binary indicator for certain variables
                    if target == 'RR':
                        df[f'Rain_Binary_Lag_{lag}'] = (df[target].shift(lag) > 0).astype(int)
                    elif target == 'ss':
                        df[f'Sunny_Binary_Lag_{lag}'] = (df[target].shift(lag) > 5).astype(int)  # Sunny if >5 hours
        
        # Add rainfall streak features (consecutive days with/without rain)
        if 'RR' in df.columns:
            df['RainToday'] = (df['RR'] > 0).astype(int)
            # Initialize streak counters
            df['Rain_Streak'] = 0
            df['Dry_Streak'] = 0
            
            # Calculate rain and dry streaks
            streak = 0
            for i in range(len(df)):
                if i == 0:
                    if df.iloc[i]['RainToday'] == 1:
                        streak = 1
                        df.loc[df.index[i], 'Rain_Streak'] = streak
                    else:
                        streak = 1
                        df.loc[df.index[i], 'Dry_Streak'] = streak
                else:
                    if df.iloc[i]['RainToday'] == 1:
                        if df.iloc[i-1]['RainToday'] == 1:
                            streak += 1
                        else:
                            streak = 1
                        df.loc[df.index[i], 'Rain_Streak'] = streak
                    else:
                        if df.iloc[i-1]['RainToday'] == 0:
                            streak += 1
                        else:
                            streak = 1
                        df.loc[df.index[i], 'Dry_Streak'] = streak
        
        # Handle any remaining NaN values in derived features using the same strategy
        derived_features = ['Temp_Range', 'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index']
        for feature in derived_features:
            if feature in df.columns and df[feature].isna().any():
                print(f"Handling NaN values in derived feature '{feature}': {df[feature].isna().sum()} values")
                # Forward fill, backward fill, then 5-day rolling mean, then median
                df[feature] = df[feature].fillna(method='ffill')
                df[feature] = df[feature].fillna(method='bfill')
                
                if df[feature].isna().any():
                    rolling_mean_5d = df[feature].rolling(window=5, center=True, min_periods=1).mean()
                    df[feature] = df[feature].fillna(rolling_mean_5d)
                
                if df[feature].isna().any():
                    df[feature] = df[feature].fillna(df[feature].median())
        
        # Handle NaN values in lag and rolling features
        lag_rolling_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in 
                               [f'{target}_Lag_' for target in TARGET_COLUMNS] + 
                               [f'{target}_Rolling_' for target in TARGET_COLUMNS] + 
                               ['Rain_Binary_Lag_', 'Sunny_Binary_Lag_', 'RH_Rolling_'])]
        for feature in lag_rolling_features:
            if df[feature].isna().any():
                # For lag features, we can only use forward fill and median (backward fill would introduce future data)
                if 'Lag_' in feature:
                    df[feature] = df[feature].fillna(method='ffill')
                    if df[feature].isna().any():
                        df[feature] = df[feature].fillna(df[feature].median())
                # For rolling features, use the same comprehensive strategy
                else:
                    df[feature] = df[feature].fillna(method='ffill')
                    df[feature] = df[feature].fillna(method='bfill')
                    if df[feature].isna().any():
                        rolling_mean_5d = df[feature].rolling(window=5, center=True, min_periods=1).mean()
                        df[feature] = df[feature].fillna(rolling_mean_5d)
                    if df[feature].isna().any():
                        df[feature] = df[feature].fillna(df[feature].median())
        
        # Update feature columns with new features that were successfully created
        # Exclude target variables from features
        base_features = [
            'Tn', 'Tx', 'RH_avg', 'ss', 'ff_x', 'ddd_x',  # Basic measurements (excluding targets)
            'Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range',
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'DayOfYear_sin', 'DayOfYear_cos',
            'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index',
            'Rain_Streak', 'Dry_Streak'
        ]
        
        # Filter out target variables from base features
        self.feature_columns = [f for f in base_features if f not in TARGET_COLUMNS and f in df.columns]
        
        # Add rolling and lag features if they exist and don't have NaN values
        for col in df.columns:
            if (any(col.startswith(prefix) for prefix in 
                   [f'{target}_Rolling_' for target in TARGET_COLUMNS] + 
                   [f'{target}_Lag_' for target in TARGET_COLUMNS] + 
                   ['Rain_Binary_Lag_', 'Sunny_Binary_Lag_', 'RH_Rolling_']) 
                and not df[col].isna().any() and col not in TARGET_COLUMNS):
                self.feature_columns.append(col)
        
        # Print final stats for all target variables
        print("\nTarget Variables Statistics After Processing:")
        for target in TARGET_COLUMNS:
            if target in df.columns:
                print(f"\n{target}:")
                print(f"  Mean: {df[target].mean():.2f}")
                print(f"  Median: {df[target].median():.2f}")
                print(f"  Std Dev: {df[target].std():.2f}")
                print(f"  Min: {df[target].min():.2f}")
                print(f"  Max: {df[target].max():.2f}")
        
        # Final feature list
        print(f"\nFinal feature list ({len(self.feature_columns)} features):")
        print(', '.join(self.feature_columns))
        
        return df 