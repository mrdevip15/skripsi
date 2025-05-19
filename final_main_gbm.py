import pandas as pd
import numpy as np
import logging
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
from sklearn.model_selection import GridSearchCV, KFold
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'gbm_prediction.log')),
        logging.StreamHandler()
    ]
)

class GBMWeatherPredictor:
    def __init__(self):
        self.model = None
        self.multi_day_models = {}  # For storing models for different forecast horizons
        self.feature_columns = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x']
        self.target_column = 'RR'
        self.scaler = StandardScaler()
        self.forecast_days = 5  # Number of days to forecast ahead
        self.cv_folds = 5  # Number of cross-validation folds
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def plot_outliers(self, df, save_path, title="Rainfall Distribution"):
        """Plot rainfall distribution to visualize outliers"""
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.subplot(121)
        plt.hist(df[self.target_column].dropna(), bins=50)
        plt.axvline(x=100, color='r', linestyle='--', label='Outlier Threshold')
        plt.title(title)
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot box plot
        plt.subplot(122)
        sns.boxplot(y=df[self.target_column].dropna())
        plt.title('Rainfall Box Plot')
        plt.ylabel('Rainfall (mm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

    def preprocess_data(self, df):
        """Enhanced preprocessing with outlier removal and better error handling"""
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            logging.info(f"Initial dataset size: {initial_size}")
            
            # Convert date
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            
            # Replace 8888 with NaN in RR column
            df['RR'] = df['RR'].replace(8888, np.nan)
            
            # Plot original distribution
            self.plot_outliers(df, self.run_dir, "Original Rainfall Distribution")
            
            # Remove outliers (RR > 100)
            outliers = df[df['RR'] > 100].copy()
            df = df[df['RR'] <= 100].copy()
            
            # Log outlier information
            if len(outliers) > 0:
                logging.info(f"\nOutliers removed: {len(outliers)} rows")
                logging.info(f"Dates with extreme rainfall (>100mm):")
                for _, row in outliers.iterrows():
                    logging.info(f"Date: {row['Tanggal']}, Rainfall: {row['RR']}mm")
            
            # Plot distribution after outlier removal
            self.plot_outliers(df, self.run_dir, "Rainfall Distribution After Outlier Removal")
            
            # Handle missing values
            # For each column, impute missing values with the median of that column
            for col in df.columns:
                if df[col].dtype != 'datetime64[ns]' and df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            
            # Convert wind direction to numeric
            wind_dir_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            # Handle wind direction in ddd_car column
            if 'ddd_car' in df.columns:
                df['ddd_car_deg'] = df['ddd_car'].apply(
                    lambda x: wind_dir_map.get(x.strip(), np.nan) if isinstance(x, str) else np.nan
                )
                # Fill NaN values in ddd_x with values from ddd_car_deg
                df['ddd_x'] = pd.to_numeric(df['ddd_x'], errors='coerce')
                df.loc[df['ddd_x'].isna(), 'ddd_x'] = df.loc[df['ddd_x'].isna(), 'ddd_car_deg']
            
            # Add basic features that won't introduce NaN values
            df['Month'] = df['Tanggal'].dt.month
            df['Day'] = df['Tanggal'].dt.day
            df['DayOfWeek'] = df['Tanggal'].dt.dayofweek
            df['DayOfYear'] = df['Tanggal'].dt.dayofyear
            df['Season'] = (df['Month'] % 12 + 3) // 3
            df['Temp_Range'] = df['Tx'] - df['Tn']
            
            # Add cyclical encoding of time features (better for capturing seasonality)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
            df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
            df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
            df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
            df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
            
            # Add interaction terms known to be important in meteorology
            df['Temp_Humidity'] = df['Tavg'] * df['RH_avg']
            df['Temp_Range_RH'] = df['Temp_Range'] * df['RH_avg']
            
            # Adding weather physics-based features
            # Simplified dew point calculation
            df['Dew_Point'] = df['Tavg'] - ((100 - df['RH_avg']) / 5)
            
            # Simplified heat index
            df['Heat_Index'] = df['Tavg'] + 0.05 * df['RH_avg']
            
            # Add rolling features with careful handling of NaN values
            windows = [3, 7, 14]  # Adding 2-week window
            for window in windows:
                # Rolling mean with min_periods=1 to avoid NaN
                df[f'RR_Rolling_Mean_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).mean()
                df[f'RR_Rolling_Std_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).std()
                df[f'Tavg_Rolling_Mean_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
            
            # Add lag features (more lags with increasing significance for rainfall prediction)
            for lag in [1, 2, 3, 5, 7, 14]:
                df[f'RR_Lag_{lag}'] = df[self.target_column].shift(lag)
                df[f'Tavg_Lag_{lag}'] = df['Tavg'].shift(lag)
                # Binary rain indicator (was it raining on that day?)
                df[f'Rain_Binary_Lag_{lag}'] = (df[self.target_column].shift(lag) > 0).astype(int)
            
            # Add rainfall streak features (consecutive days with/without rain)
            df['RainToday'] = (df[self.target_column] > 0).astype(int)
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
            
            # Drop rows with NaN values that couldn't be imputed
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Log any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                logging.info(f"\nRows dropped due to NaN values: {len(df_before_drop) - len(df)}")
            
            # Update feature columns with new features that were successfully created
            self.feature_columns = [
                'Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x',
                'Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range',
                'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                'Temp_Humidity', 'Temp_Range_RH', 'Dew_Point', 'Heat_Index',
                'Rain_Streak', 'Dry_Streak'
            ]
            
            # Add rolling and lag features if they exist and don't have NaN values
            for col in df.columns:
                if (col.startswith(('RR_Rolling', 'Tavg_Rolling', 'RH_Rolling', 'RR_Lag', 'Tavg_Lag', 'Rain_Binary_Lag')) 
                    and not df[col].isna().any()):
                    self.feature_columns.append(col)
            
            # Log final stats
            logging.info(f"\nFinal dataset size: {len(df)}")
            logging.info(f"Total rows removed: {initial_size - len(df)}")
            logging.info("\nRainfall Statistics After Processing:")
            logging.info(f"Mean: {df[self.target_column].mean():.2f}mm")
            logging.info(f"Median: {df[self.target_column].median():.2f}mm")
            logging.info(f"Std Dev: {df[self.target_column].std():.2f}mm")
            logging.info(f"Max: {df[self.target_column].max():.2f}mm")
            
            # Check no NaN values remain
            nan_counts = df.isna().sum()
            if nan_counts.sum() > 0:
                logging.warning("\nWARNING: NaN values still exist in the dataset:")
                logging.warning(nan_counts[nan_counts > 0])
            else:
                logging.info("\nNo NaN values remain in the dataset.")
            
            # Final feature list
            logging.info(f"\nFinal feature list ({len(self.feature_columns)} features):")
            logging.info(', '.join(self.feature_columns))
            
            return df
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def plot_feature_correlations(self, df):
        """Plot correlation matrix of features"""
        # Use only a subset of features if there are too many
        if len(self.feature_columns) > 15:
            # Calculate correlation with target
            correlations = {}
            for feature in self.feature_columns:
                correlations[feature] = abs(np.corrcoef(df[feature], df[self.target_column])[0, 1])
            
            # Get the top 15 most correlated features
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_features = [f[0] for f in top_features]
            features_to_plot = selected_features + [self.target_column]
            logging.info(f"\nPlotting correlations for top 15 features: {', '.join(selected_features)}")
        else:
            features_to_plot = self.feature_columns + [self.target_column]
        
        # Create correlation matrix
        corr = df[features_to_plot].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_correlations.png'))
        plt.close()

    def plot_feature_distributions(self, df):
        """Plot distribution of top features"""
        # Select top features by correlation with target
        correlations = {}
        for feature in self.feature_columns:
            correlations[feature] = abs(np.corrcoef(df[feature], df[self.target_column])[0, 1])
        
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:9]
        selected_features = [f[0] for f in top_features] + [self.target_column]
        
        # Plot
        n_features = len(selected_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, feature in enumerate(selected_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[feature], kde=True)
            plt.title(f'{feature} Distribution (corr: {correlations.get(feature, 0):.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_distributions.png'))
        plt.close()

    def plot_seasonal_patterns(self, df):
        """Plot seasonal patterns of rainfall"""
        plt.figure(figsize=(12, 6))
        monthly_avg = df.groupby('Month')[self.target_column].mean()
        monthly_avg.plot(kind='bar')
        plt.title('Average Rainfall by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Rainfall (mm)')
        plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'seasonal_patterns.png'))
        plt.close()

    def plot_predictions(self, dates, actual, predicted):
        """Plot actual vs predicted rainfall"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, actual, marker='o', linestyle='-', label='Actual', alpha=0.7)
        plt.plot(dates, predicted, marker='x', linestyle='-', label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Rainfall')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'predictions.png'))
        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        if self.model is None:
            logging.warning("Model not trained yet, cannot plot feature importance.")
            return
            
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [self.feature_columns[i] for i in indices]
        sorted_importance = feature_importance[indices]
        
        # Plot top 20 features or all if less than 20
        plt.figure(figsize=(12, 8))
        n_features = min(20, len(self.feature_columns))
        plt.barh(range(n_features), sorted_importance[:n_features], align='center')
        plt.yticks(range(n_features), sorted_feature_names[:n_features])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_importance.png'))
        plt.close()
        
        # Also compute permutation importance
        logging.info("Computing permutation importance (this may take a while)...")
        X_train = self.X_train_scaled
        y_train = self.y_train
        
        result = permutation_importance(
            self.model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_importance = result.importances_mean
        indices = np.argsort(perm_importance)[::-1]
        sorted_feature_names = [self.feature_columns[i] for i in indices]
        sorted_importance = perm_importance[indices]
        
        # Plot top 20 features by permutation importance
        plt.figure(figsize=(12, 8))
        n_features = min(20, len(self.feature_columns))
        plt.barh(range(n_features), sorted_importance[:n_features], align='center')
        plt.yticks(range(n_features), sorted_feature_names[:n_features])
        plt.xlabel('Permutation Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance (Permutation Method)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'permutation_importance.png'))
        plt.close()
        
        # Log feature importance
        logging.info("\nFeature Importance (top 10):")
        for i in range(min(10, len(sorted_feature_names))):
            logging.info(f"{sorted_feature_names[i]}: {sorted_importance[i]:.4f}")

    def save_metrics(self, metrics):
        """Save evaluation metrics to a file"""
        with open(os.path.join(self.run_dir, 'metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    
    def prepare_multi_day_dataset(self, df):
        """Prepare dataset for multi-day forecasting using past 5 days data"""
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
        logging.info(f"\nInitial data shape: {df_copy.shape}")
        logging.info(f"Initial NaN counts:\n{df_copy.isna().sum()}")
        
        # Create features from past 5 days for each feature
        for feature in self.feature_columns:
            for i in range(1, 6):  # Past 5 days
                df_copy[f'{feature}_past_{i}d'] = df_copy[feature].shift(i)
        
        # Create specialized weekly patterns
        # Weekly rainfall patterns (same day of week)
        for i in range(1, 5):  # Past 4 weeks of same day
            df_copy[f'RR_SameDay_{i}wk'] = df_copy[self.target_column].shift(i*7)
        
        # Add rate of change features
        df_copy['RR_1d_change'] = df_copy[self.target_column].diff()
        df_copy['RR_3d_change'] = df_copy[self.target_column] - df_copy[self.target_column].shift(3)
        df_copy['RR_7d_change'] = df_copy[self.target_column] - df_copy[self.target_column].shift(7)
        
        # Add volatility indicators
        df_copy['RR_volatility_3d'] = df_copy[self.target_column].rolling(window=3).std()
        df_copy['RR_volatility_7d'] = df_copy[self.target_column].rolling(window=7).std()
        
        # Add rainfall seasonality features
        month_avg = df_copy.groupby(df_copy['Tanggal'].dt.month)[self.target_column].transform('mean')
        df_copy['RR_vs_MonthAvg'] = df_copy[self.target_column] / month_avg
        
        # Add cumulative rain features
        df_copy['RR_cum_3d'] = df_copy[self.target_column].rolling(window=3).sum()
        df_copy['RR_cum_7d'] = df_copy[self.target_column].rolling(window=7).sum()
        
        # Weather pattern indicators
        # Is it currently in a rainy pattern?
        df_copy['RainPattern_3d'] = ((df_copy[self.target_column] > 0) & 
                                   (df_copy[self.target_column].shift(1) > 0) & 
                                   (df_copy[self.target_column].shift(2) > 0)).astype(int)
        # Is it currently in a dry pattern?
        df_copy['DryPattern_3d'] = ((df_copy[self.target_column] == 0) & 
                                  (df_copy[self.target_column].shift(1) == 0) & 
                                  (df_copy[self.target_column].shift(2) == 0)).astype(int)
        
        # Create more complex interaction features
        df_copy['RH_Tavg_RR_Lag1'] = df_copy['RH_avg'] * df_copy['Tavg'] * df_copy[f'RR_Lag_1']
        df_copy['Wind_RH_Lag1'] = df_copy['ff_avg'] * df_copy['RH_avg'] * df_copy[f'RR_Lag_1']
        
        # Log state after creating features
        logging.info(f"\nAfter adding all features - shape: {df_copy.shape}")
        logging.info(f"NaN counts after features:\n{df_copy.isna().sum()}")
        
        # Update feature columns to include all new features
        historical_features = []
        for feature in self.feature_columns:
            historical_features.extend([f'{feature}_past_{i}d' for i in range(1, 6)])
            
        # Add new feature names
        new_features = [
            'RR_SameDay_1wk', 'RR_SameDay_2wk', 'RR_SameDay_3wk', 'RR_SameDay_4wk',
            'RR_1d_change', 'RR_3d_change', 'RR_7d_change',
            'RR_volatility_3d', 'RR_volatility_7d',
            'RR_vs_MonthAvg', 'RR_cum_3d', 'RR_cum_7d',
            'RainPattern_3d', 'DryPattern_3d',
            'RH_Tavg_RR_Lag1', 'Wind_RH_Lag1'
        ]
        
        # Combine current, historical, and new features
        self.extended_features = self.feature_columns + historical_features + new_features
        
        # Handle missing values in all features
        for feature in self.extended_features:
            if feature in df_copy.columns:
                # First try forward fill
                df_copy[feature] = df_copy[feature].fillna(method='ffill')
                # Then backward fill any remaining NaNs
                df_copy[feature] = df_copy[feature].fillna(method='bfill')
                # Finally, if any NaNs remain, fill with column median
                if df_copy[feature].isna().any():
                    df_copy[feature] = df_copy[feature].fillna(df_copy[feature].median())
        
        # Log state after handling missing values
        logging.info(f"\nAfter handling missing values - shape: {df_copy.shape}")
        logging.info(f"NaN counts after handling missing values:\n{df_copy.isna().sum()}")
        
        # First include today (day 0) - no need to shift the target
        day_df = df_copy[['Tanggal'] + self.extended_features].copy()
        day_df[f'Future_RR_0d'] = df_copy[self.target_column]
        day_df = day_df.dropna()  # Remove any remaining NaN rows
        multi_day_data[0] = day_df.copy()
        
        # For each forecast day (1 to forecast_days)
        for day in range(1, self.forecast_days + 1):
            # Shift target variable to create future target
            future_target = df_copy[self.target_column].shift(-day)
            
            # Create dataset with current features, historical features, date column, and future target
            day_df = df_copy[['Tanggal'] + self.extended_features].copy()
            day_df[f'Future_RR_{day}d'] = future_target
            
            # Drop rows with NaN
            day_df = day_df.dropna()
            
            # Add seasonal stratification for better model training
            # Create season indicator for model training
            day_df['Season_Indicator'] = day_df['Tanggal'].dt.month.apply(
                lambda m: 1 if m in [12, 1, 2] else  # Winter
                         2 if m in [3, 4, 5] else    # Spring
                         3 if m in [6, 7, 8] else    # Summer
                         4                           # Fall
            )
            
            # Log info about the dataset for this day
            logging.info(f"\nDay {day} dataset - shape: {len(day_df)}")
            logging.info(f"Number of NaN values in day {day} dataset: {day_df.isna().sum().sum()}")
            
            multi_day_data[day] = day_df
            
            # Log final dataset size for this day
            logging.info(f"Final size of day {day} dataset: {len(multi_day_data[day])}")
        
        return multi_day_data

    def plot_feature_importance_extended(self, feature_importance, day):
        """Plot feature importance showing only top 10 most important features"""
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [self.extended_features[i] for i in indices]
        sorted_importance = feature_importance[indices]
        
        # Take only top 10 features
        top_n = 10
        top_features = sorted_feature_names[:top_n]
        top_importance = sorted_importance[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importance)
        plt.yticks(y_pos, top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for {day}-Day Ahead Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f'feature_importance_day{day}.png'))
        plt.close()
        
        # Log top 10 features
        logging.info(f"\nTop {top_n} most important features for {day}-day ahead prediction:")
        for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
            logging.info(f"{i+1}. {feature}: {importance:.4f}")

    def train_multi_day_models(self, multi_day_data):
        """Train separate advanced models for each forecast day using ensemble approach"""
        results = {}
        predictions = {}  # To store predictions for cascading forecast
        
        # More lightweight and efficient model configurations
        base_models = {
            'gbm': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=4,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=N_JOBS
            ),
            'ridge': Ridge(
                alpha=1.0, 
                solver='auto',
                random_state=42
            )
        }
        
        # Process each forecast day
        for day in range(self.forecast_days + 1):  # Include day 0
            logging.info(f"\n--- Training models for {day}-day ahead prediction ---")
            
            if day not in multi_day_data:
                logging.warning(f"No data available for day {day}, skipping")
                continue
                
            day_df = multi_day_data[day].copy()
            
            # Add specialized temporal features for different forecast horizons
            if day > 0:
                self._add_specialized_temporal_features(day_df, day)
                
                # Add cascading prediction features for n > 1 day forecasts
                if day > 1 and day-1 in predictions:
                    self._add_cascading_predictions(day_df, predictions[day-1], day)
            
            # Use fewer features for faster training and select appropriate features for horizon
            features_to_use = self._select_features_for_horizon(day, self.extended_features)
            
            # Add any new features created
            for col in day_df.columns:
                if col.startswith(('fourier_', 'wavelet_', 'cascade_pred_')) and col not in features_to_use:
                    features_to_use.append(col)
            
            if len(features_to_use) > 100:  # If too many features, select subset
                # For longer horizons, prioritize cyclical and seasonal features
                if day >= 3:
                    # Ensure key temporal features are included
                    priority_features = [f for f in features_to_use if 
                                       any(p in f for p in ['Month', 'Season', 'fourier_', 'wavelet_', 
                                                          'cascade_pred_', 'SameDay', '_cum'])]
                    # Get remaining features up to 100 total
                    remaining_count = 100 - len(priority_features)
                    other_features = [f for f in features_to_use if f not in priority_features]
                    selected_features = self._select_most_important_features(
                        day_df, other_features, f'Future_RR_{day}d', remaining_count)
                    features_to_use = priority_features + selected_features
                else:
                    features_to_use = self._select_most_important_features(
                        day_df, features_to_use, f'Future_RR_{day}d', 100)
            
            logging.info(f"Using {len(features_to_use)} features for day {day} prediction")
            
            # Split data
            train_size = int(0.8 * len(day_df))
            
            # Keep dates in a separate variable
            train_dates = day_df.iloc[:train_size]['Tanggal']
            test_dates = day_df.iloc[train_size:]['Tanggal']
            
            # Extract features and target
            X_train = day_df.iloc[:train_size][features_to_use]
            y_train = day_df.iloc[:train_size][f'Future_RR_{day}d']
            X_test = day_df.iloc[train_size:][features_to_use]
            y_test = day_df.iloc[train_size:][f'Future_RR_{day}d']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Use fewer CV folds for faster training
            cv_folds = 3 if day == 0 else 2  # Reduced from 5
            
            # Apply more regularization for longer horizons to prevent overfitting
            if day > 1:
                # Adjust regularization parameters for longer horizons
                base_models['gbm'].max_depth = max(3, 5 - day//2)  # Reduce tree depth
                base_models['rf'].max_depth = max(6, 10 - day)  # Reduce tree depth
                base_models['ridge'].alpha = 1.0 + day * 0.5  # Increase regularization
            
            # Initialize cross-validation
            cv_scores = {model_name: [] for model_name in base_models.keys()}
            cv_predictions = {model_name: np.zeros(len(X_test)) for model_name in base_models.keys()}
            
            # Show progress during training
            logging.info(f"Training models with {cv_folds}-fold CV...")
            model_names = list(base_models.keys())
            
            # Train each model with timeout protection
            for model_name in model_names:
                logging.info(f"Training {model_name} model...")
                base_model = base_models[model_name]
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                fold_predictions = []
                
                # Process each fold with progress tracking
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
                    logging.info(f"  Processing fold {fold+1}/{cv_folds}...")
                    start_time = time.time()
                    
                    # Split data for this fold
                    X_fold_train = X_train_scaled[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train_scaled[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    try:
                        # Clone and train model with timeout protection
                        model = clone(base_model)
                        model.fit(X_fold_train, y_fold_train)
                        
                        # Validate
                        val_pred = model.predict(X_fold_val)
                        r2 = r2_score(y_fold_val, val_pred)
                        cv_scores[model_name].append(r2)
                        
                        # Predict on test set
                        fold_predictions.append(model.predict(X_test_scaled))
                        
                        elapsed = time.time() - start_time
                        logging.info(f"    Fold R²: {r2:.4f} (took {elapsed:.1f}s)")
                        
                    except Exception as e:
                        logging.error(f"Error in fold {fold+1}: {str(e)}")
                        # Use average of previous folds or 0 if no previous folds
                        if fold_predictions:
                            fold_predictions.append(np.mean(fold_predictions, axis=0))
                        else:
                            fold_predictions.append(np.zeros(len(X_test_scaled)))
                        cv_scores[model_name].append(0.0)
                
                # Average predictions across folds
                if fold_predictions:
                    cv_predictions[model_name] = np.mean(fold_predictions, axis=0)
                    logging.info(f"{model_name} CV R² scores: {np.mean(cv_scores[model_name]):.4f} ± {np.std(cv_scores[model_name]):.4f}")
                else:
                    logging.warning(f"No valid predictions for {model_name}, using zeros")
                    cv_predictions[model_name] = np.zeros(len(X_test))
            
            # Calculate ensemble weights based on CV performance
            weights = self._calculate_ensemble_weights(cv_scores)
            
            # Create weighted ensemble prediction
            ensemble_pred = np.zeros(len(X_test))
            for model_name in base_models.keys():
                if model_name in weights:  # Check if model has a weight
                    ensemble_pred += weights[model_name] * cv_predictions[model_name]
            
            # Store full dataset predictions for cascading
            all_features = day_df[features_to_use]
            all_scaled = scaler.transform(all_features)
            all_preds = {}
            
            # Make predictions on full dataset
            for model_name, model in base_models.items():
                if model_name in weights:
                    try:
                        all_preds[model_name] = model.predict(all_scaled)
                    except:
                        # If model failed, use zeros
                        all_preds[model_name] = np.zeros(len(all_scaled))
            
            # Create full dataset ensemble prediction
            full_ensemble_pred = np.zeros(len(all_scaled))
            for model_name in all_preds:
                if model_name in weights:
                    full_ensemble_pred += weights[model_name] * all_preds[model_name]
            
            # Store for cascading predictions
            predictions[day] = {
                'dates': day_df['Tanggal'],
                'predictions': full_ensemble_pred
            }
            
            # Calculate prediction standard deviation for uncertainty estimation
            pred_std = np.std([cv_predictions[model_name] for model_name in base_models.keys() 
                            if model_name in cv_predictions], axis=0)
            
            # Calculate final metrics
            ensemble_metrics = {
                'mse': mean_squared_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'r2': r2_score(y_test, ensemble_pred)
            }
            
            logging.info(f"Ensemble metrics - R²: {ensemble_metrics['r2']:.4f}, RMSE: {ensemble_metrics['rmse']:.4f}")
            
            # Store results
            results[day] = {
                'test_dates': test_dates,
                'y_test': y_test,
                'y_pred': ensemble_pred,
                'pred_std': pred_std,
                'metrics': ensemble_metrics,
                'features_used': features_to_use
            }
            
            # Save model data
            self.multi_day_models[day] = {
                'model_name': 'ensemble',
                'models': base_models,
                'weights': weights,
                'scaler': scaler,
                'features_used': features_to_use,
                'pred_std': pred_std
            }
            
            # Plot results
            self._plot_prediction_with_error_bars(test_dates, y_test, ensemble_pred, 
                                                ensemble_pred - 1.96 * pred_std,
                                                ensemble_pred + 1.96 * pred_std,
                                                day, ensemble_metrics)
            
        return results
        
    def _add_specialized_temporal_features(self, df, forecast_day):
        """Add specialized temporal features for medium and long-term forecasts"""
        # Add Fourier features for seasonal patterns - especially useful for 3+ day forecasts
        for period in [365.25, 180, 90, 30]:  # Annual, semi-annual, quarterly, monthly
            # Create sin and cos features for the given period
            df[f'fourier_sin_{period}'] = np.sin(2 * np.pi * df['Tanggal'].dt.dayofyear / period)
            df[f'fourier_cos_{period}'] = np.cos(2 * np.pi * df['Tanggal'].dt.dayofyear / period)
            
            # For longer forecasts, add harmonics
            if forecast_day >= 3:
                for harmonic in [2, 3]:
                    df[f'fourier_sin_{period}_h{harmonic}'] = np.sin(2 * np.pi * harmonic * df['Tanggal'].dt.dayofyear / period)
                    df[f'fourier_cos_{period}_h{harmonic}'] = np.cos(2 * np.pi * harmonic * df['Tanggal'].dt.dayofyear / period)
        
        # Add specialized long-term features for 3+ day forecasts
        if forecast_day >= 3:
            # Long-term rainfall running statistics (14-30 days)
            if 'RR' in df.columns:
                # Monthly climatology - what's normal for this time of year
                df['monthly_rain_avg'] = df.groupby(df['Tanggal'].dt.month)['RR'].transform('mean')
                df['monthly_rain_std'] = df.groupby(df['Tanggal'].dt.month)['RR'].transform('std')
                
                # Seasonal patterns
                df['seasonal_pattern'] = df['RR'].rolling(window=30, min_periods=1).mean()
                
                # Rain frequency in recent period (% of days with rain)
                df['rain_frequency_30d'] = df['RR'].rolling(window=30, min_periods=1).apply(
                    lambda x: np.sum(x > 0) / len(x))
        
        logging.info(f"Added specialized temporal features for {forecast_day}-day ahead prediction")
        
    def _add_cascading_predictions(self, df, prev_day_preds, forecast_day):
        """Add previous day predictions as features for current day"""
        # Create a DataFrame with dates and predictions
        prev_preds_df = pd.DataFrame({
            'Tanggal': prev_day_preds['dates'],
            f'cascade_pred_{forecast_day-1}d': prev_day_preds['predictions']
        })
        
        # Merge with current dataframe
        df_merged = pd.merge(df, prev_preds_df, on='Tanggal', how='left')
        
        # Copy merged columns back to original dataframe
        for col in prev_preds_df.columns:
            if col != 'Tanggal':
                df[col] = df_merged[col]
        
        # Fill any NaN values with appropriate stats
        if f'cascade_pred_{forecast_day-1}d' in df.columns:
            # Forward fill first
            df[f'cascade_pred_{forecast_day-1}d'] = df[f'cascade_pred_{forecast_day-1}d'].fillna(method='ffill')
            # Then use mean for any remaining NaN
            df[f'cascade_pred_{forecast_day-1}d'] = df[f'cascade_pred_{forecast_day-1}d'].fillna(
                df[f'cascade_pred_{forecast_day-1}d'].mean())
        
        # Add derived features from cascaded predictions
        if f'cascade_pred_{forecast_day-1}d' in df.columns:
            # Binary rain prediction from previous day
            df[f'cascade_rain_binary_{forecast_day-1}d'] = (df[f'cascade_pred_{forecast_day-1}d'] > 0.5).astype(int)
            
            # If we have more than one previous day prediction
            if forecast_day > 2 and f'cascade_pred_{forecast_day-2}d' in df.columns:
                # Rate of change in predictions
                df[f'cascade_pred_change'] = df[f'cascade_pred_{forecast_day-1}d'] - df[f'cascade_pred_{forecast_day-2}d']
                
                # Trend continuation feature (is trend continuing in same direction)
                if f'cascade_pred_change_prev' in df.columns:
                    df['cascade_trend_continues'] = np.sign(df[f'cascade_pred_change']) == np.sign(df['cascade_pred_change_prev'])
                
                # Store current change for next day
                df['cascade_pred_change_prev'] = df[f'cascade_pred_change']
        
        logging.info(f"Added cascading prediction features for {forecast_day}-day ahead prediction")

    def _select_features_for_horizon(self, day, all_features):
        """Select appropriate features based on forecast horizon"""
        if day == 0:  # Today's prediction
            return all_features
        
        # Base features that are always included
        base_features = [f for f in self.feature_columns if not f.startswith(('RR_Lag', 'Tavg_Lag'))]
        
        # Features specific to short-term prediction (1-2 days)
        if day <= 2:
            short_term_patterns = ['_past_1d', '_past_2d', 'Lag_1', 'Lag_2', 'Rolling_Mean_3d',
                                'RainPattern', 'Temp_', 'RH_', 'Rain_Streak', 'DryPattern',
                                'volatility_3d', 'cum_3d']
            
            specific_features = [f for f in all_features if 
                               any(pattern in f for pattern in short_term_patterns)]
        else:
            # Features specific to medium-term prediction (3-5 days)
            medium_term_patterns = ['Month', 'Season', 'DayOfYear', 'Month_sin', 'Month_cos',
                                  'DayOfYear_sin', 'DayOfYear_cos', 'Rolling_Mean_7d',
                                  'Rolling_Mean_14d', 'SameDay', 'cum_7d', 'volatility_7d']
            
            # Add fourier patterns if available
            fourier_patterns = ['fourier_', 'wavelet_', 'monthly_rain', 'seasonal_pattern',
                              'rain_frequency']
            
            specific_features = [f for f in all_features if 
                               any(pattern in f for pattern in medium_term_patterns + fourier_patterns)]
            
            # For longer term, remove most recent lags which become less relevant
            specific_features = [f for f in specific_features if 'Lag_1' not in f and 'past_1d' not in f]
        
        # Add cascade prediction features if available
        cascade_features = [f for f in all_features if f.startswith('cascade_')]
        
        return list(set(base_features + specific_features + cascade_features))

    def _calculate_ensemble_weights(self, cv_scores):
        """Calculate weights for ensemble based on cross-validation performance"""
        mean_scores = {model: np.mean(scores) for model, scores in cv_scores.items()}
        min_score = min(mean_scores.values())
        max_score = max(mean_scores.values())
        
        # Normalize scores to [0.1, 1] range
        if max_score > min_score:
            weights = {model: 0.1 + 0.9 * (score - min_score) / (max_score - min_score)
                      for model, score in mean_scores.items()}
        else:
            # If all scores are equal, use equal weights
            weights = {model: 1.0 / len(cv_scores) for model in cv_scores.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {model: w / total_weight for model, w in weights.items()}
        
        return weights

    def _plot_prediction_with_error_bars(self, dates, actual, predicted, lower, upper, day, metrics):
        """Create detailed plot of predictions with error bars and metrics"""
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(dates, actual, 'o-', color='blue', label='Actual', alpha=0.7, markersize=4)
        
        # Plot predicted values
        plt.plot(dates, predicted, 'x-', color='red', label='Predicted', alpha=0.7, markersize=4)
        
        # Add prediction intervals
        plt.fill_between(dates, lower, upper, color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Add metrics as text
        metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                       f"MAE: {metrics['mae']:.2f}\n"
                       f"R²: {metrics['r2']:.2f}")
        plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        # Identify days with large errors
        errors = np.abs(actual - predicted)
        large_error_threshold = np.percentile(errors, 90)  # Top 10% of errors
        large_error_indices = errors > large_error_threshold
        
        if np.any(large_error_indices):
            plt.scatter(dates[large_error_indices], actual[large_error_indices],
                      color='purple', s=100, marker='o', alpha=0.7,
                      label='Large Error Points')
            
            # Add annotations for large errors
            for date, act, pred in zip(dates[large_error_indices], 
                                     actual[large_error_indices], 
                                     predicted[large_error_indices]):
                error = abs(act - pred)
                plt.annotate(f'Error: {error:.1f}mm',
                           xy=(date, max(act, pred)),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        # Customize plot
        day_label = "Today" if day == 0 else f"{day}-Day Ahead"
        plt.title(f'{day_label} Rainfall Prediction with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.run_dir, f'detailed_prediction_day{day}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_multi_day_predictions(self, results):
        """Plot predictions for multiple forecast horizons with metrics"""
        # +1 is for including day 0 (today)
        plt.figure(figsize=(15, (self.forecast_days + 1) * 2))
        
        # First plot day 0 (today's) prediction
        for day in range(0, self.forecast_days + 1):
            if day not in results:
                continue
                
            plt.subplot(self.forecast_days + 1, 1, day + 1)
            
            dates = results[day]['test_dates']
            actual = results[day]['y_test']
            predicted = results[day]['y_pred']
            metrics = results[day]['metrics']
            
            plt.plot(dates, actual, marker='o', markersize=4, linestyle='-', label='Actual', alpha=0.7)
            plt.plot(dates, predicted, marker='x', markersize=4, linestyle='-', label='Predicted', alpha=0.7)
            
            # Format and add metrics to the plot
            metrics_text = f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}"
            
            # Special title for day 0
            if day == 0:
                plt.title(f'Today\'s Prediction - {metrics_text}')
            else:
                plt.title(f'{day}-Day Ahead Prediction - {metrics_text}')
                
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if day == self.forecast_days:  # Only show dates on bottom subplot
                plt.xlabel('Date')
                plt.xticks(rotation=45)
            else:
                plt.xticks([])  # Hide x ticks for non-bottom subplots
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_day_predictions.png'))
        plt.close()

    def plot_prediction_intervals(self, results):
        """Plot prediction intervals for each day ahead forecast with metrics"""
        for day in range(1, self.forecast_days + 1):
            if day not in results:
                continue
                
            dates = results[day]['test_dates']
            actual = results[day]['y_test']
            predicted = results[day]['y_pred']
            std_dev = results[day]['pred_std']
            metrics = results[day]['metrics']
            
            plt.figure(figsize=(15, 6))
            
            # Plot actual and predicted
            plt.plot(dates, actual, 'o-', label='Actual', alpha=0.7)
            plt.plot(dates, predicted, 'x-', label='Predicted', alpha=0.7)
            
            # Plot confidence intervals (95%)
            upper = predicted + 1.96 * std_dev
            lower = predicted - 1.96 * std_dev
            lower = np.maximum(lower, 0)  # Rainfall can't be negative
            
            plt.fill_between(dates, lower, upper, alpha=0.2, color='gray', label='95% Confidence Interval')
            
            # Format and add metrics to the plot
            metrics_text = f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}"
            plt.title(f'{day}-Day Ahead Prediction with Confidence Intervals - {metrics_text}')
            plt.xlabel('Date')
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, f'prediction_intervals_day{day}.png'))
            plt.close()

    def save_multi_day_models(self):
        """Save multi-day models to disk"""
        for day, model_data in self.multi_day_models.items():
            model_dir = os.path.join(self.models_dir, f'day_{day}')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save ensemble information
            ensemble_path = os.path.join(model_dir, 'ensemble_info.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump({
                    'weights': model_data['weights'],
                    'features_used': model_data['features_used'],
                    'pred_std': model_data['pred_std']
                }, f)
            
            # Save individual models
            for model_name, model in model_data['models'].items():
                model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
                joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(model_data['scaler'], scaler_path)
            
            logging.info(f"Saved ensemble model for {day}-day ahead prediction in {model_dir}")
            
    def load_multi_day_models(self):
        """Load saved multi-day models from disk"""
        self.multi_day_models = {}
        
        # Check each day directory
        for day in range(self.forecast_days + 1):  # Include day 0
            model_dir = os.path.join(self.models_dir, f'day_{day}')
            if not os.path.exists(model_dir):
                logging.warning(f"No saved model found for day {day}")
                continue
                
            # Load ensemble information
            ensemble_path = os.path.join(model_dir, 'ensemble_info.pkl')
            if not os.path.exists(ensemble_path):
                logging.warning(f"No ensemble info found for day {day}")
                continue
                
            with open(ensemble_path, 'rb') as f:
                ensemble_info = pickle.load(f)
            
            # Initialize model data structure
            self.multi_day_models[day] = {
                'model_name': 'ensemble',
                'models': {},
                'weights': ensemble_info['weights'],
                'features_used': ensemble_info['features_used'],
                'pred_std': ensemble_info['pred_std'],
                'scaler': None
            }
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.multi_day_models[day]['scaler'] = joblib.load(scaler_path)
            
            # Load individual models
            for model_name in ['gbm', 'rf', 'ridge']:
                model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.multi_day_models[day]['models'][model_name] = joblib.load(model_path)
            
            logging.info(f"Loaded ensemble model for {day}-day ahead prediction")
            
        return len(self.multi_day_models) > 0

    def train_and_evaluate(self, df):
        """Train and evaluate the GBM model on the processed dataset"""
        try:
            # Prepare feature matrix and target vector
            X = df[self.feature_columns]
            y = df[self.target_column]
            
            # Log shape of the feature matrix
            logging.info(f"\nFeature matrix shape: {X.shape}")
            
            # Split into training and testing sets
            train_size = int(0.8 * len(df))
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]
            
            # Save for later use in permutation importance
            self.X_train = X_train
            self.y_train = y_train
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            logging.info("\nTraining Gradient Boosting model...")
            
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            )
            
            self.model.fit(self.X_train_scaled, y_train)
            
            # Make predictions on the test set
            logging.info("Making predictions...")
            y_pred = self.model.predict(X_test_scaled)
            
            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            logging.info("\nModel evaluation metrics:")
            logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
            logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
            logging.info(f"R-squared (R²): {r2:.4f}")
            
            # Save metrics
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            self.save_metrics(metrics)
            
            # Plot results
            self.plot_feature_importance()
            
            # Plot predictions with correct date handling
            test_dates = df.iloc[train_size:]['Tanggal'].values
            self.plot_predictions(test_dates, y_test.values, y_pred)
            
            # Multi-day forecasting
            logging.info("\nPreparing multi-day forecasting dataset...")
            multi_day_data = self.prepare_multi_day_dataset(df)
            
            logging.info("\nTraining multi-day forecasting models...")
            multi_day_results = self.train_multi_day_models(multi_day_data)
            
            # Plot multi-day prediction results
            self.plot_multi_day_predictions(multi_day_results)
            self.plot_prediction_intervals(multi_day_results)
            
            # Save multi-day models
            self.save_multi_day_models()
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in model training and evaluation: {str(e)}")
            raise

    def save_model(self):
        """Save the trained model, scaler, and feature list to disk"""
        if self.model is None:
            logging.warning("Model not trained yet, nothing to save.")
            return
            
        model_path = os.path.join(self.models_dir, 'gbm_model.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        features_path = os.path.join(self.models_dir, 'feature_columns.pkl')
        
        # Save model, scaler, and feature list
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
            
        logging.info(f"\nModel saved to {model_path}")
        logging.info(f"Scaler saved to {scaler_path}")
        logging.info(f"Feature list saved to {features_path}")

    def _select_most_important_features(self, df, features, target_col, max_features=100):
        """Select the most important features based on correlation with target"""
        correlations = []
        for feature in features:
            if feature in df.columns:
                corr = abs(np.corrcoef(df[feature].values, df[target_col].values)[0, 1])
                if not np.isnan(corr):
                    correlations.append((feature, corr))
        
        # Sort by correlation and take top features
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in correlations[:max_features]]
        
        logging.info(f"Selected {len(selected)} most correlated features from {len(features)} total")
        return selected

def main():
    try:
        # Load dataset
        logging.info("Loading dataset...")
        df = pd.read_csv('makassar.csv')
        
        # Initialize predictor
        predictor = GBMWeatherPredictor()
        
        # Preprocess data
        logging.info("Preprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Plot additional visualizations
        predictor.plot_feature_correlations(processed_df)
        predictor.plot_feature_distributions(processed_df)
        predictor.plot_seasonal_patterns(processed_df)
        
        # Train and evaluate
        logging.info("Training and evaluating model...")
        predictor.train_and_evaluate(processed_df)
        
        # Save the trained model
        predictor.save_model()
        
        logging.info("Process completed successfully.")
        
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise
    finally:
        # Clean up matplotlib resources
        plt.close('all')

if __name__ == "__main__":
    main() 