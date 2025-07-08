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

class GBMWeatherPredictor:
    def __init__(self):
        self.model = None
        self.multi_day_models = {}  # For storing models for different forecast horizons
        self.multi_target_models = {}  # For storing models for different target variables
        
        # Define multiple target variables
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)',
            'Tavg': 'Average Temperature (°C)',
            'ddd_car': 'Wind Direction',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # Reduced set of most important features (excluding target variables from features)
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
        
        self.scaler = StandardScaler()
        self.target_scalers = {}  # Separate scalers for each target
        self.forecast_days = 5  # Number of days to forecast ahead
        self.cv_folds = 5  # Number of time series splits
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def plot_target_distributions(self, df, save_path, title_prefix="Target Variable Distribution"):
        """Plot distributions of all target variables"""
        n_targets = len(self.target_columns)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, target in enumerate(self.target_columns, 1):
            plt.subplot(n_rows, n_cols, i)
            if target in df.columns:
                plt.hist(df[target].dropna(), bins=50, alpha=0.7)
                plt.title(f'{self.target_names[target]}')
                plt.xlabel(target)
                plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{title_prefix.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_rainfall_distribution(self, df, save_path, title="Rainfall Distribution"):
        """Plot rainfall distribution - updated to use new method"""
        self.plot_target_distributions(df, save_path, title)

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
                        
                        print(f"  - Forward fill: {filled_by_ffill} values")
                        print(f"  - Backward fill: {filled_by_bfill} values") 
                        print(f"  - 5-day rolling mean: {filled_by_rolling} values")
                        print(f"  - Successfully interpolated all {special_count} special values in '{col}'")
            
            # Plot original distribution
            self.plot_target_distributions(df, self.run_dir, "Original Target Variables Distribution")
            
            # Convert wind direction to numeric
            wind_dir_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            # Handle wind direction in ddd_car column - with better error handling
            if 'ddd_car' in df.columns:
                # First clean up the strings (remove trailing spaces)
                if df['ddd_car'].dtype == object:  # Check if it's a string column
                    df['ddd_car'] = df['ddd_car'].astype(str).str.strip()
                    
                # Convert to degrees using mapping, but keep original for prediction
                df['ddd_car_numeric'] = df['ddd_car'].apply(
                    lambda x: wind_dir_map.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
                )
                
                # For prediction purposes, we'll use the numeric version
                # But we need to handle the original ddd_car values properly
                # Convert ddd_car to numeric if it contains degree values
                df['ddd_car'] = pd.to_numeric(df['ddd_car'], errors='coerce')
                # Fill NaN values with the mapped numeric values
                df['ddd_car'] = df['ddd_car'].fillna(df['ddd_car_numeric'])
                
                # Handle ddd_x column - ensure it's numeric first
                if 'ddd_x' in df.columns:
                    # Try to convert to numeric, coercing errors to NaN
                    df['ddd_x'] = pd.to_numeric(df['ddd_x'], errors='coerce')
                    # Fill NaN values in ddd_x with values from ddd_car
                    df.loc[df['ddd_x'].isna(), 'ddd_x'] = df.loc[df['ddd_x'].isna(), 'ddd_car']
            
            # Handle missing values in other columns (excluding special values already handled)
            for col in df.columns:
                if df[col].dtype != 'datetime64[ns]' and df[col].isna().any():
                    # Handle numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use similar interpolation strategy as for special values
                        print(f"Handling missing values in column '{col}': {df[col].isna().sum()} NaN values")
                        
                        # Forward fill
                        df[col] = df[col].fillna(method='ffill')
                        # Backward fill
                        df[col] = df[col].fillna(method='bfill')
                        
                        # If any NaNs remain, use 5-day rolling mean
                        if df[col].isna().any():
                            rolling_mean_5d = df[col].rolling(window=5, center=True, min_periods=1).mean()
                            df[col] = df[col].fillna(rolling_mean_5d)
                        
                        # Final fallback: use median
                        if df[col].isna().any():
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                            
                    # Handle string/object columns
                    elif df[col].dtype == object:
                        # For string columns, use mode (most common value)
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(mode_val)
            
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
            for target in self.target_columns:
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
                for target in self.target_columns:
                    if target in df.columns:
                        # Rolling mean with min_periods=1 to avoid NaN
                        df[f'{target}_Rolling_Mean_{window}d'] = df[target].rolling(window=window, min_periods=1).mean()
                        df[f'{target}_Rolling_Std_{window}d'] = df[target].rolling(window=window, min_periods=1).std()
                
                # Also add rolling features for other important variables
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
            
            # Add lag features for all target variables (more lags with increasing significance)
            for lag in [1, 2, 3, 5, 7, 14]:
                for target in self.target_columns:
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
                                   [f'{target}_Lag_' for target in self.target_columns] + 
                                   [f'{target}_Rolling_' for target in self.target_columns] + 
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
            
            # Final check: drop any rows that still have NaN values (should be very few if any)
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Print any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                print(f"\nRows dropped due to remaining NaN values: {len(df_before_drop) - len(df)}")
                print("Note: These should be minimal after comprehensive interpolation")
            
            # Update feature columns with new features that were successfully created
            # Exclude target variables from features
            base_features = [
                'Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x',  # Basic measurements (excluding targets)
                'Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range',
                'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index',
                'Rain_Streak', 'Dry_Streak'
            ]
            
            # Filter out target variables from base features
            self.feature_columns = [f for f in base_features if f not in self.target_columns and f in df.columns]
            
            # Add rolling and lag features if they exist and don't have NaN values
            rolling_lag_features_added = []
            for col in df.columns:
                if (any(col.startswith(prefix) for prefix in 
                       [f'{target}_Rolling_' for target in self.target_columns] + 
                       [f'{target}_Lag_' for target in self.target_columns] + 
                       ['Rain_Binary_Lag_', 'Sunny_Binary_Lag_', 'RH_Rolling_']) 
                    and not df[col].isna().any() and col not in self.target_columns):
                    # Exclude ALL rolling and lag features to see impact of engineered features
                    # if not col.endswith('Rolling_Mean_3d'):
                    #     self.feature_columns.append(col)
                    #     rolling_lag_features_added.append(col)
                    pass  # Skip all rolling and lag features
            
            print(f"\nRolling and lag features excluded: All rolling and lag features removed")
            print(f"Only using basic features and engineered features")
            
            # Add engineered features that might have been missed
            engineered_features = [
                'Temp_Range', 'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index',
                'Rain_Streak', 'Dry_Streak', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
                'DayOfYear_sin', 'DayOfYear_cos'
            ]

            engineered_features_available = [f for f in engineered_features if f in df.columns]
            engineered_features_added = []

            for feature in engineered_features:
                if feature in df.columns and feature not in self.feature_columns and feature not in self.target_columns:
                    if not df[feature].isna().any():
                        self.feature_columns.append(feature)
                        engineered_features_added.append(feature)
                        print(f"Added engineered feature: {feature}")

            print(f"\nEngineered features summary:")
            print(f"Available in dataset: {len(engineered_features_available)}/{len(engineered_features)}")
            print(f"Added to model: {len(engineered_features_added)}")
            print(f"Available features: {', '.join(engineered_features_available)}")
            print(f"Added features: {', '.join(engineered_features_added)}")
            
            # Print final stats for all target variables
            print(f"\nFinal dataset size: {len(df)}")
            print(f"Total rows removed: {initial_size - len(df)}")
            print("\nTarget Variables Statistics After Processing:")
            for target in self.target_columns:
                if target in df.columns:
                    print(f"\n{self.target_names[target]} ({target}):")
                    print(f"  Mean: {df[target].mean():.2f}")
                    print(f"  Median: {df[target].median():.2f}")
                    print(f"  Std Dev: {df[target].std():.2f}")
                    print(f"  Min: {df[target].min():.2f}")
                    print(f"  Max: {df[target].max():.2f}")
            
            # Check no NaN values remain
            nan_counts = df.isna().sum()
            if nan_counts.sum() > 0:
                print("\nWARNING: NaN values still exist in the dataset:")
                print(nan_counts[nan_counts > 0])
            else:
                print("\nNo NaN values remain in the dataset - interpolation successful!")
            
            # Print feature importance debugging info
            print(f"\nFeature engineering summary:")
            print(f"Total features available: {len(df.columns)}")
            print(f"Target variables: {self.target_columns}")
            print(f"Features used in model: {len(self.feature_columns)}")
            print(f"Engineered features included: {[f for f in self.feature_columns if f in engineered_features]}")
            
            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def plot_feature_correlations(self, df):
        """Plot correlation matrix of features with all target variables"""
        # Use only a subset of features if there are too many
        if len(self.feature_columns) > 12:  # Reduced to make room for multiple targets
            # Calculate correlation with all targets
            feature_importance = {}
            for feature in self.feature_columns:
                # Calculate average absolute correlation with all targets
                correlations = []
                for target in self.target_columns:
                    if target in df.columns:
                        corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                feature_importance[feature] = np.mean(correlations) if correlations else 0
            
            # Get the top 12 most correlated features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
            selected_features = [f[0] for f in top_features]
            features_to_plot = selected_features + self.target_columns
            print(f"\nPlotting correlations for top 12 features: {', '.join(selected_features)}")
        else:
            features_to_plot = self.feature_columns + self.target_columns
        
        # Filter features that exist in the dataframe
        features_to_plot = [f for f in features_to_plot if f in df.columns]
        
        # Create correlation matrix
        corr = df[features_to_plot].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Feature Correlations with Target Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_correlations.png'))
        plt.close()

    def plot_feature_distributions(self, df):
        """Plot distribution of top features and all target variables"""
        # Select top features by average correlation with all targets
        feature_importance = {}
        for feature in self.feature_columns:
            correlations = []
            for target in self.target_columns:
                if target in df.columns:
                    corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
            feature_importance[feature] = np.mean(correlations) if correlations else 0
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:6]
        selected_features = [f[0] for f in top_features]
        
        # Combine top features with all target variables
        features_to_plot = selected_features + [t for t in self.target_columns if t in df.columns]
        
        # Plot
        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, feature in enumerate(features_to_plot, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[feature], kde=True)
            
            # Add correlation info for features, target name for targets
            if feature in self.target_columns:
                plt.title(f'{self.target_names.get(feature, feature)} (Target)')
            else:
                avg_corr = feature_importance.get(feature, 0)
                plt.title(f'{feature} (avg corr: {avg_corr:.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_distributions.png'))
        plt.close()

    def plot_seasonal_patterns(self, df):
        """Plot seasonal patterns of all target variables"""
        n_targets = len([t for t in self.target_columns if t in df.columns])
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        plot_idx = 1
        
        for target in self.target_columns:
            if target in df.columns:
                plt.subplot(n_rows, n_cols, plot_idx)
                monthly_avg = df.groupby('Month')[target].mean()
                monthly_avg.plot(kind='bar')
                plt.title(f'Average {self.target_names[target]} by Month')
                plt.xlabel('Month')
                plt.ylabel(self.target_names[target])
                plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
                plot_idx += 1
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'seasonal_patterns.png'))
        plt.close()

    def plot_data_splits(self, df):
        """Plot data splits for each target parameter in separate figures with splits arranged vertically"""
        # Prepare data
        df_sorted = df.sort_values('Tanggal')
        dates = df_sorted['Tanggal']
        total_samples = len(df_sorted)
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Colors for train and validation data
        train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
        val_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']    # Light Red, Teal, Light Blue, Light Green, Yellow
        
        # Create separate figure for each target parameter
        for target in self.target_columns:
            if target not in df_sorted.columns:
                continue
                
            # Create figure with subplots for each split with proper spacing
            fig, axes = plt.subplots(self.cv_folds, 1, figsize=(15, 5*self.cv_folds), 
                                   gridspec_kw={'hspace': 0.3})
            if self.cv_folds == 1:
                axes = [axes]
            
            target_data = df_sorted[target]
            
            # Plot each split in separate subplot
            for split_idx, (train_idx, val_idx) in enumerate(tscv.split(df_sorted)):
                if split_idx < len(train_colors):
                    train_color = train_colors[split_idx]
                    val_color = val_colors[split_idx]
                    ax = axes[split_idx]
                    
                    # Plot all data points in background
                    ax.plot(dates, target_data, 'k-', alpha=0.2, linewidth=0.5, label='All Data')
                    
                    # Plot training data for this split
                    train_dates = dates.iloc[train_idx]
                    train_data = target_data.iloc[train_idx]
                    ax.plot(train_dates, train_data, color=train_color, linewidth=2.5, 
                           label=f'Split {split_idx+1} Train', alpha=0.9)
                    
                    # Plot validation data for this split
                    val_dates = dates.iloc[val_idx]
                    val_data = target_data.iloc[val_idx]
                    ax.plot(val_dates, val_data, color=val_color, linewidth=2.5, 
                           label=f'Split {split_idx+1} Val', alpha=0.9)
                    
                    # Customize subplot with proper spacing
                    ax.set_title(f'{self.target_names[target]} - Split {split_idx+1}', 
                               fontsize=12, fontweight='bold', pad=20)
                    ax.set_ylabel(self.target_names[target], fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))
                    
                    # Format x-axis
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add split statistics with better positioning
                    train_size = len(train_idx)
                    val_size = len(val_idx)
                    split_info = f"Train: {train_size} samples\nVal: {val_size} samples\n"
                    split_info += f"Train period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}\n"
                    split_info += f"Val period: {val_dates.min().strftime('%Y-%m-%d')} to {val_dates.max().strftime('%Y-%m-%d')}"
                    
                    ax.text(0.02, 0.95, split_info, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                           facecolor='white', alpha=0.9), fontsize=8)
            
            # No overall title - removed as requested
            
            # Adjust layout without suptitle
            plt.tight_layout()
            
            # Save individual figure for this target
            target_safe_name = target.replace(' ', '_').lower()
            plt.savefig(os.path.join(self.run_dir, f'data_splits_{target_safe_name}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Data splits visualization for {self.target_names[target]} saved")
        
        # Print split statistics
        print(f"\nTime Series Split Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Number of splits: {self.cv_folds}")
        
        for split_idx, (train_idx, val_idx) in enumerate(tscv.split(df_sorted)):
            train_size = len(train_idx)
            val_size = len(val_idx)
            print(f"Split {split_idx+1}: Train={train_size} samples, Val={val_size} samples")

    def plot_predictions(self, dates, actual, predicted, target_name='Target'):
        """Plot actual vs predicted values for a specific target"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, actual, marker='o', linestyle='-', label='Actual', alpha=0.7)
        plt.plot(dates, predicted, marker='x', linestyle='-', label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted {target_name}')
        plt.xlabel('Date')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Improved filename sanitization - remove all problematic characters
        target_safe_name = (target_name.replace(' ', '_')
                           .replace('(', '')
                           .replace(')', '')
                           .replace('/', '_')
                           .replace('\\', '_')
                           .replace(':', '_')
                           .replace('*', '_')
                           .replace('?', '_')
                           .replace('"', '_')
                           .replace('<', '_')
                           .replace('>', '_')
                           .replace('|', '_')
                           .lower())
        
        plt.savefig(os.path.join(self.run_dir, f'predictions_{target_safe_name}.png'))
        plt.close()

    def plot_multi_target_feature_importance(self, target_models):
        """Plot feature importance for all target variables"""
        n_targets = len(target_models)
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        
        for i, (target, model_data) in enumerate(target_models.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Get feature importance
            feature_importance = model_data['model'].feature_importances_
            
            # Sort features by importance (show top 10 instead of 5)
            indices = np.argsort(feature_importance)[::-1][:10]
            sorted_feature_names = [self.feature_columns[idx] for idx in indices]
            sorted_importance = feature_importance[indices]
            
            # Plot horizontal bar chart
            plt.barh(range(len(sorted_importance)), sorted_importance)
            plt.yticks(range(len(sorted_importance)), sorted_feature_names)
            plt.xlabel('Importance')
            plt.title(f'Top 10 Features - {self.target_names[target]}')
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Add debugging info
            print(f"\nFeature importance for {self.target_names[target]}:")
            for j, (name, importance) in enumerate(zip(sorted_feature_names, sorted_importance)):
                print(f"  {j+1:2d}. {name:25s}: {importance:.4f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_target_feature_importance.png'))
        plt.close()
        
        # Also create a comprehensive feature importance analysis
        self.plot_comprehensive_feature_importance(target_models)

    def plot_comprehensive_feature_importance(self, target_models):
        """Create a comprehensive feature importance analysis"""
        # Calculate average importance across all targets
        all_importances = {}
        for target, model_data in target_models.items():
            feature_importance = model_data['model'].feature_importances_
            for i, feature in enumerate(self.feature_columns):
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(feature_importance[i])
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature, importances in all_importances.items():
            avg_importance[feature] = np.mean(importances)
        
        # Sort by average importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Plot comprehensive feature importance
        plt.figure(figsize=(12, 8))
        features, importances = zip(*sorted_features[:15])  # Top 15 features
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Average Importance Across All Targets')
        plt.title('Comprehensive Feature Importance Analysis')
        plt.gca().invert_yaxis()
        
        # Add color coding for engineered features
        engineered_features = [
            'Temp_Range', 'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index',
            'Rain_Streak', 'Dry_Streak', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos'
        ]
        
        colors = ['red' if f in engineered_features else 'blue' for f in features]
        for i, (bar, color) in enumerate(zip(plt.gca().patches, colors)):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'comprehensive_feature_importance.png'))
        plt.close()
        
        # Print comprehensive analysis
        print(f"\n=== Comprehensive Feature Importance Analysis ===")
        print(f"Total features analyzed: {len(sorted_features)}")
        print(f"Engineered features in top 15: {[f for f, _ in sorted_features[:15] if f in engineered_features]}")
        
        for i, (feature, importance) in enumerate(sorted_features[:15], 1):
            feature_type = "ENGINEERED" if feature in engineered_features else "ORIGINAL"
            print(f"{i:2d}. {feature:25s}: {importance:.4f} ({feature_type})")

    def plot_feature_importance(self):
        """Plot feature importance from the trained model - updated for multi-target"""
        if not self.multi_target_models:
            print("Models not trained yet, cannot plot feature importance.")
            return
            
        self.plot_multi_target_feature_importance(self.multi_target_models)

    def save_metrics(self, metrics, target_name=''):
        """Save evaluation metrics to a file"""
        filename = f'metrics_{target_name}.txt' if target_name else 'metrics.txt'
        with open(os.path.join(self.run_dir, filename), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    
    def prepare_multi_day_dataset(self, df):
        """Prepare dataset for multi-day forecasting - updated for multiple targets"""
        multi_day_data = {}
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure Tanggal is a datetime index or column
        if 'Tanggal' not in df_copy.columns:
            df_copy.reset_index(inplace=True)
            df_copy.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        # Sort data by date to ensure correct ordering
        df_copy = df_copy.sort_values('Tanggal')
        
        # Log initial state
        print(f"\nInitial data shape: {df_copy.shape}")
        
        # Handle missing values in base features first
        for feature in self.feature_columns:
            if df_copy[feature].isna().any():
                # Use forward fill first
                df_copy[feature] = df_copy[feature].fillna(method='ffill')
                # Then backward fill
                df_copy[feature] = df_copy[feature].fillna(method='bfill')
                # If any NaNs remain, use median
                df_copy[feature] = df_copy[feature].fillna(df_copy[feature].median())
        
        # For each target variable, create datasets for each forecast day
        for target in self.target_columns:
            if target not in df_copy.columns:
                continue
                
            multi_day_data[target] = {}
            
            # For day 0 (today) - current day prediction
            day_df = df_copy[['Tanggal'] + self.feature_columns].copy()
            day_df[f'Future_{target}_0d'] = df_copy[target]
            day_df = day_df.dropna()  # Remove any remaining NaN rows
            multi_day_data[target][0] = day_df.copy()
            
            # For each forecast day (1 to forecast_days)
            for day in range(1, self.forecast_days + 1):
                print(f"\nPreparing {target} dataset for {day}-day ahead prediction")
                
                # Create dataset with base features
                day_df = df_copy[['Tanggal'] + self.feature_columns].copy()
                
                # Shift target variable to create future target
                day_df[f'Future_{target}_{day}d'] = df_copy[target].shift(-day)
                
                # Drop rows with NaN in target (these will be the last 'day' rows)
                day_df = day_df.dropna(subset=[f'Future_{target}_{day}d'])
                
                # Add seasonal stratification for better model training
                day_df['Season_Indicator'] = day_df['Tanggal'].dt.month.apply(
                    lambda m: 1 if m in [12, 1, 2] else  # Winter
                             2 if m in [3, 4, 5] else    # Spring
                             3 if m in [6, 7, 8] else    # Summer
                             4                           # Fall
                )
                
                # Log dataset info
                print(f"  {target} Day {day} dataset shape: {day_df.shape}")
                print(f"  Date range: {day_df['Tanggal'].min()} to {day_df['Tanggal'].max()}")
                
                multi_day_data[target][day] = day_df
        
        return multi_day_data

    def plot_predictions_for_day(self, dates, actual, predicted, day, metrics, title_suffix=""):
        """Plot predictions for a specific forecast day and target"""
        plt.figure(figsize=(15, 6))
        
        # Plot actual and predicted values
        plt.plot(dates, actual, 'o-', label='Actual', alpha=0.7, markersize=4)
        plt.plot(dates, predicted, 'x-', label='Predicted', alpha=0.7, markersize=4)
        
        # Add metrics as text
        metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                       f"MAE: {metrics['mae']:.2f}\n"
                       f"R²: {metrics['r2']:.2f}")
        plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        # Customize plot
        day_label = "Today" if day == 0 else f"{day}-Day Ahead"
        full_title = f'{day_label} Prediction'
        if title_suffix:
            full_title = f'{title_suffix}'
            
        plt.title(full_title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Improved filename sanitization - remove all problematic characters
        safe_title = (title_suffix.replace(' ', '_')
                     .replace('(', '')
                     .replace(')', '')
                     .replace('/', '_')
                     .replace('\\', '_')
                     .replace(':', '_')
                     .replace('*', '_')
                     .replace('?', '_')
                     .replace('"', '_')
                     .replace('<', '_')
                     .replace('>', '_')
                     .replace('|', '_')
                     .lower())
        
        plt.savefig(os.path.join(self.run_dir, f'prediction_{safe_title}_day_{day}.png'))
        plt.close()

    def train_multi_day_models(self, multi_day_data):
        """Train separate GBM models for each target and forecast day"""
        results = {}
        
        # Define base model configurations for different targets
        model_configs = {
            'RR': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                min_samples_split=5, min_samples_leaf=4, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'ss': GradientBoostingRegressor(
                n_estimators=180, learning_rate=0.06, max_depth=4,
                min_samples_split=6, min_samples_leaf=5, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'Tavg': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                min_samples_split=8, min_samples_leaf=6, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'ddd_car': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                min_samples_split=8, min_samples_leaf=6, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'ff_avg': GradientBoostingRegressor(
                n_estimators=160, learning_rate=0.07, max_depth=4,
                min_samples_split=7, min_samples_leaf=5, subsample=0.8,
                max_features='sqrt', random_state=42
            )
        }
        
        # Process each target variable
        for target in self.target_columns:
            if target not in multi_day_data:
                print(f"Skipping {target} - no multi-day data available")
                continue
                
            print(f"\n=== Training Multi-Day Models for {self.target_names[target]} ===")
            results[target] = {}
            
            # Process each forecast day for this target
            for day, day_df in multi_day_data[target].items():
                print(f"\n--- {target}: {day}-day ahead prediction ---")
                
                # Use base features for all predictions
                features_to_use = self.feature_columns
                print(f"Using {len(features_to_use)} features for {target} day {day} prediction")
                
                # Sort by date to ensure proper temporal split
                day_df = day_df.sort_values('Tanggal')
                
                # Split data while preserving temporal order
                train_size = int(0.8 * len(day_df))
                
                # Keep dates in a separate variable
                train_dates = day_df.iloc[:train_size]['Tanggal']
                test_dates = day_df.iloc[train_size:]['Tanggal']
                
                # Extract features and target
                X_train = day_df.iloc[:train_size][features_to_use]
                y_train = day_df.iloc[:train_size][f'Future_{target}_{day}d']
                X_test = day_df.iloc[train_size:][features_to_use]
                y_test = day_df.iloc[train_size:][f'Future_{target}_{day}d']
                
                # Log data splits info
                print(f"Training data: {len(X_train)} samples from {train_dates.min()} to {train_dates.max()}")
                print(f"Testing data: {len(X_test)} samples from {test_dates.min()} to {test_dates.max()}")
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use fewer CV folds for faster training
                cv_folds = 3 if day == 0 else 2
                
                # Initialize cross-validation
                cv_scores = []
                
                # Show progress during training
                print(f"Training {target} GBM model with {cv_folds} time series splits...")
                
                # Train model with time series cross-validation
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                fold_predictions = []
                
                # Get base model for this target
                base_model = model_configs.get(target, model_configs['RR'])  # Default to RR config
                
                # Process each split
                for split_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
                    print(f"  Processing split {split_idx+1}/{cv_folds}...")
                    start_time = time.time()
                    
                    # Split data for this split
                    X_fold_train = X_train_scaled[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train_scaled[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    try:
                        # Train model
                        model = clone(base_model)
                        model.fit(X_fold_train, y_fold_train)
                        
                        # Validate
                        val_pred = model.predict(X_fold_val)
                        r2 = r2_score(y_fold_val, val_pred)
                        cv_scores.append(r2)
                        
                        # Predict on test set
                        test_pred = model.predict(X_test_scaled)
                        
                        # Apply target-specific constraints
                        if target == 'RR':
                            test_pred = np.maximum(test_pred, 0)
                        elif target == 'ss':
                            test_pred = np.clip(test_pred, 0, 24)
                        elif target == 'ff_avg':
                            test_pred = np.maximum(test_pred, 0)
                        elif target == 'ddd_car':
                            test_pred = np.clip(test_pred % 360, 0, 360)
                        
                        fold_predictions.append(test_pred)
                        
                        elapsed = time.time() - start_time
                        print(f"    Split R²: {r2:.4f} (took {elapsed:.1f}s)")
                        
                    except Exception as e:
                        print(f"Error in split {split_idx+1}: {str(e)}")
                        if fold_predictions:
                            fold_predictions.append(np.mean(fold_predictions, axis=0))
                        else:
                            fold_predictions.append(np.zeros(len(X_test_scaled)))
                        cv_scores.append(0.0)
                
                # Average predictions across splits
                if fold_predictions:
                    final_predictions = np.mean(fold_predictions, axis=0)
                    print(f"{target} GBM Time Series CV R² scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                else:
                    print(f"No valid predictions for {target}, using zeros")
                    final_predictions = np.zeros(len(X_test))
                
                # Apply final constraints
                if target == 'RR':
                    final_predictions = np.maximum(final_predictions, 0)
                elif target == 'ss':
                    final_predictions = np.clip(final_predictions, 0, 24)
                elif target == 'ff_avg':
                    final_predictions = np.maximum(final_predictions, 0)
                elif target == 'ddd_car':
                    final_predictions = np.clip(final_predictions % 360, 0, 360)
                
                # Calculate prediction standard deviation for uncertainty estimation
                pred_std = np.std(fold_predictions, axis=0) if len(fold_predictions) > 1 else np.zeros_like(final_predictions)
                
                # Calculate final metrics
                final_metrics = {
                    'mse': mean_squared_error(y_test, final_predictions),
                    'rmse': np.sqrt(mean_squared_error(y_test, final_predictions)),
                    'mae': mean_absolute_error(y_test, final_predictions),
                    'r2': r2_score(y_test, final_predictions)
                }
                
                print(f"Final {target} metrics - R²: {final_metrics['r2']:.4f}, RMSE: {final_metrics['rmse']:.4f}")
                
                # Store results
                results[target][day] = {
                    'test_dates': test_dates,
                    'y_test': y_test,
                    'y_pred': final_predictions,
                    'pred_std': pred_std,
                    'metrics': final_metrics,
                    'features_used': features_to_use
                }
                
                # Store model data
                if target not in self.multi_day_models:
                    self.multi_day_models[target] = {}
                    
                self.multi_day_models[target][day] = {
                    'model': base_model,
                    'scaler': scaler,
                    'features_used': features_to_use,
                    'pred_std': pred_std
                }
                
                # Plot results for this target and day
                day_label = "Today" if day == 0 else f"{day}-Day Ahead"
                target_name = self.target_names[target]
                self.plot_predictions_for_day(test_dates, y_test, final_predictions, day, final_metrics, f"{target_name} {day_label}")
                
        return results

    def plot_multi_day_predictions(self, results):
        """Plot predictions for multiple forecast horizons and targets with metrics"""
        
        # Create separate plots for each target variable
        for target in self.target_columns:
            if target not in results:
                continue
                
            target_results = results[target]
            n_days = len(target_results)
            
            if n_days == 0:
                continue
                
            plt.figure(figsize=(15, n_days * 2))
            
            # Plot each forecast day for this target
            for i, day in enumerate(sorted(target_results.keys())):
                plt.subplot(n_days, 1, i + 1)
                
                dates = target_results[day]['test_dates']
                actual = target_results[day]['y_test']
                predicted = target_results[day]['y_pred']
                metrics = target_results[day]['metrics']
                
                plt.plot(dates, actual, marker='o', markersize=4, linestyle='-', label='Actual', alpha=0.7)
                plt.plot(dates, predicted, marker='x', markersize=4, linestyle='-', label='Predicted', alpha=0.7)
                
                # Format and add metrics to the plot
                metrics_text = f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}"
                
                # Special title for day 0
                if day == 0:
                    plt.title(f'{self.target_names[target]} - Today\'s Prediction - {metrics_text}')
                else:
                    plt.title(f'{self.target_names[target]} - {day}-Day Ahead Prediction - {metrics_text}')
                    
                plt.ylabel(self.target_names[target])
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if i == n_days - 1:  # Only show dates on bottom subplot
                    plt.xlabel('Date')
                    plt.xticks(rotation=45)
                else:
                    plt.xticks([])  # Hide x ticks for non-bottom subplots
            
            plt.tight_layout()
            target_name_safe = target.replace(' ', '_').lower()
            plt.savefig(os.path.join(self.run_dir, f'multi_day_predictions_{target_name_safe}.png'))
            plt.close()
        
        # Create a summary plot showing R² scores for all targets and days
        self.plot_multi_day_summary(results)

    def plot_multi_day_summary(self, results):
        """Plot summary of R² scores for all targets and forecast days"""
        # Prepare data for summary plot
        targets = []
        days = []
        r2_scores = []
        
        for target, target_results in results.items():
            for day, day_results in target_results.items():
                targets.append(self.target_names[target])
                days.append(f"Day {day}" if day > 0 else "Today")
                r2_scores.append(day_results['metrics']['r2'])
        
        if not targets:
            return
            
        # Create a pivot table-like structure for heatmap
        import pandas as pd
        summary_df = pd.DataFrame({
            'Target': targets,
            'Forecast_Day': days,
            'R2_Score': r2_scores
        })
        
        # Pivot for heatmap
        pivot_df = summary_df.pivot(index='Target', columns='Forecast_Day', values='R2_Score')
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f',
                   cbar_kws={'label': 'R² Score'})
        plt.title('Multi-Target Multi-Day Prediction Performance (R² Scores)')
        plt.xlabel('Forecast Horizon')
        plt.ylabel('Target Variable')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_day_summary_heatmap.png'))
        plt.close()
        
        # Also create a line plot showing performance degradation over forecast days
        plt.figure(figsize=(12, 6))
        
        unique_targets = list(set(targets))
        unique_days = sorted(list(set([int(d.split()[1]) if d != "Today" else 0 for d in days])))
        
        for target_name in unique_targets:
            target_key = None
            for key, name in self.target_names.items():
                if name == target_name:
                    target_key = key
                    break
            
            if target_key and target_key in results:
                target_r2_scores = []
                target_days = []
                
                for day in unique_days:
                    if day in results[target_key]:
                        target_r2_scores.append(results[target_key][day]['metrics']['r2'])
                        target_days.append(day)
                
                plt.plot(target_days, target_r2_scores, marker='o', label=target_name, linewidth=2)
        
        plt.xlabel('Forecast Day')
        plt.ylabel('R² Score')
        plt.title('Prediction Performance vs Forecast Horizon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'performance_vs_forecast_horizon.png'))
        plt.close()

    def save_multi_target_models(self):
        """Save all multi-target models to disk"""
        # Save main models
        for target, model_data in self.multi_target_models.items():
            target_dir = os.path.join(self.models_dir, f'target_{target}')
            os.makedirs(target_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(target_dir, 'gbm_model.pkl')
            joblib.dump(model_data['model'], model_path)
            
            # Save scaler if exists
            if model_data['scaler'] is not None:
                scaler_path = os.path.join(target_dir, 'target_scaler.pkl')
                joblib.dump(model_data['scaler'], scaler_path)
            
            print(f"Saved {self.target_names[target]} model in {target_dir}")
        
        # Save multi-day models
        for target, target_models in self.multi_day_models.items():
            for day, model_data in target_models.items():
                model_dir = os.path.join(self.models_dir, f'target_{target}', f'day_{day}')
                os.makedirs(model_dir, exist_ok=True)
                
                # Save model
                model_path = os.path.join(model_dir, 'gbm_model.pkl')
                joblib.dump(model_data['model'], model_path)
                
                # Save scaler
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
                joblib.dump(model_data['scaler'], scaler_path)
                
                # Save feature list
                features_path = os.path.join(model_dir, 'features.pkl')
                with open(features_path, 'wb') as f:
                    pickle.dump(model_data['features_used'], f)
        
        # Save main feature scaler and feature list
        main_scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, main_scaler_path)
        
        features_path = os.path.join(self.models_dir, 'feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
            
        print(f"\nSaved feature scaler and feature list in {self.models_dir}")
        print(f"Saved multi-day models for {len(self.multi_day_models)} targets, {self.forecast_days + 1} days each")

    def train_and_evaluate(self, df):
        """Train and evaluate GBM models for all target variables"""
        try:
            print(f"\n=== Training Multi-Target Weather Prediction Models ===")
            print(f"Target variables: {', '.join([self.target_names[t] for t in self.target_columns])}")
            
            # Prepare feature matrix
            X = df[self.feature_columns]
            print(f"\nFeature matrix shape: {X.shape}")
            print(f"Using {len(self.feature_columns)} features")
            
            # Split into training and testing sets (temporal split)
            train_size = int(0.8 * len(df))
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:]
            
            # Scale features once
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store for later use
            self.X_train = X_train
            self.X_train_scaled = X_train_scaled
            
            # Train separate models for each target variable
            all_metrics = {}
            
            for target in self.target_columns:
                if target not in df.columns:
                    print(f"\nSkipping {target} - not found in dataset")
                    continue
                    
                print(f"\n--- Training model for {self.target_names[target]} ({target}) ---")
                
                # Prepare target variable
                y = df[target]
                y_train = y.iloc[:train_size]
                y_test = y.iloc[train_size:]
                
                # Scale target if needed (for regression targets)
                target_scaler = StandardScaler()
                if target in ['RR', 'ss', 'ff_avg']:  # These benefit from scaling
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    self.target_scalers[target] = target_scaler
                else:
                    y_train_scaled = y_train.values
                    self.target_scalers[target] = None
                
                # Define model for this target
                if target == 'ddd_car':  # Wind direction - circular regression
                    model = GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.08,
                        max_depth=4,
                        min_samples_split=8,
                        min_samples_leaf=6,
                        subsample=0.8,
                        random_state=42
                    )
                elif target in ['RR', 'ss']:  # Rainfall and sunshine - can be zero-inflated
                    model = GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        min_samples_split=5,
                        min_samples_leaf=4,
                        subsample=0.8,
                        random_state=42
                    )
                else:  # Temperature and wind speed - more continuous
                    model = GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.06,
                        max_depth=4,
                        min_samples_split=6,
                        min_samples_leaf=5,
                        subsample=0.8,
                        random_state=42
                    )
                
                # Train the model
                print(f"Training {target} model...")
                model.fit(X_train_scaled, y_train_scaled)
                
                # Make predictions
                y_pred_scaled = model.predict(X_test_scaled)
                
                # Inverse transform if scaling was applied
                if self.target_scalers[target] is not None:
                    y_pred = self.target_scalers[target].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                else:
                    y_pred = y_pred_scaled
                
                # Apply constraints based on target type
                if target == 'RR':  # Rainfall cannot be negative
                    y_pred = np.maximum(y_pred, 0)
                elif target == 'ss':  # Sunshine hours: 0-24 hours
                    y_pred = np.clip(y_pred, 0, 24)
                elif target == 'ff_avg':  # Wind speed cannot be negative
                    y_pred = np.maximum(y_pred, 0)
                elif target == 'ddd_car':  # Wind direction: 0-360 degrees
                    y_pred = np.clip(y_pred % 360, 0, 360)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate additional metrics specific to target
                if target == 'RR':
                    # Rain-specific metrics
                    rain_days_actual = (y_test > 0).sum()
                    rain_days_predicted = (y_pred > 0).sum()
                    rain_detection_accuracy = np.mean((y_test > 0) == (y_pred > 0))
                    
                    print(f"Rain Detection Accuracy: {rain_detection_accuracy:.4f}")
                    print(f"Actual rain days: {rain_days_actual}, Predicted: {rain_days_predicted}")
                
                # Print metrics
                print(f"\n{self.target_names[target]} Model Evaluation:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
                
                # Store model and results
                self.multi_target_models[target] = {
                    'model': model,
                    'scaler': self.target_scalers[target],
                    'metrics': {
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2
                    }
                }
                
                # Save metrics for this target
                self.save_metrics(self.multi_target_models[target]['metrics'], target)
                
                # Plot predictions for this target
                test_dates = df.iloc[train_size:]['Tanggal'] if 'Tanggal' in df.columns else range(len(y_test))
                self.plot_predictions(test_dates, y_test, y_pred, self.target_names[target])
                
                # Store for summary
                all_metrics[target] = self.multi_target_models[target]['metrics']
            
            # Print summary of all models
            print(f"\n=== Multi-Target Model Summary ===")
            for target, metrics in all_metrics.items():
                print(f"{self.target_names[target]:25} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Plot comprehensive visualizations
            print(f"\nGenerating visualization plots...")
            self.plot_feature_importance()
            self.plot_feature_correlations(df)
            self.plot_feature_distributions(df)
            self.plot_seasonal_patterns(df)
            self.plot_data_splits(df)  # Add data splits visualization
            
            # Multi-day forecasting for all targets
            print(f"\n=== Multi-Day Forecasting ===")
            print("Preparing multi-day forecasting dataset...")
            multi_day_data = self.prepare_multi_day_dataset(df)
            
            print("Training multi-day forecasting models...")
            multi_day_results = self.train_multi_day_models(multi_day_data)
            
            # Plot multi-day prediction results
            self.plot_multi_day_predictions(multi_day_results)
            
            # Save all models
            self.save_multi_target_models()
            
            return all_metrics
            
        except Exception as e:
            print(f"Error in model training and evaluation: {str(e)}")
            raise

    def save_model(self):
        """Save the trained models - updated for multi-target"""
        if not self.multi_target_models:
            print("Models not trained yet, nothing to save.")
            return
            
        self.save_multi_target_models()

def main():
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv('makassar.csv')
        
        # Initialize predictor
        predictor = GBMWeatherPredictor()
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Train and evaluate - this will also generate all plots
        print("Training and evaluating model...")
        predictor.train_and_evaluate(processed_df)
        
        print("Process completed successfully.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        # Clean up matplotlib resources
        plt.close('all')

if __name__ == "__main__":
    main() 