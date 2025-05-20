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
        """Prepare dataset for multi-day forecasting"""
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
        
        # Handle missing values in base features first
        for feature in self.feature_columns:
            if df_copy[feature].isna().any():
                # Use forward fill first
                df_copy[feature] = df_copy[feature].fillna(method='ffill')
                # Then backward fill
                df_copy[feature] = df_copy[feature].fillna(method='bfill')
                # If any NaNs remain, use median
                df_copy[feature] = df_copy[feature].fillna(df_copy[feature].median())
        
        # For day 0 (today)
        day_df = df_copy[['Tanggal'] + self.feature_columns].copy()
        day_df[f'Future_RR_0d'] = df_copy[self.target_column]
        day_df = day_df.dropna()  # Remove any remaining NaN rows
        multi_day_data[0] = day_df.copy()
        
        # For each forecast day (1 to forecast_days)
        for day in range(1, self.forecast_days + 1):
            logging.info(f"\nPreparing dataset for {day}-day ahead prediction")
            
            # Create dataset with base features
            day_df = df_copy[['Tanggal'] + self.feature_columns].copy()
            
            # Shift target variable to create future target
            # Note: shift(-day) means we're looking 'day' days into the future
            day_df[f'Future_RR_{day}d'] = df_copy[self.target_column].shift(-day)
            
            # Log NaN counts before dropping
            nan_counts = day_df.isna().sum()
            logging.info(f"NaN counts before dropping:\n{nan_counts}")
            
            # Drop rows with NaN in target (these will be the last 'day' rows)
            day_df = day_df.dropna(subset=[f'Future_RR_{day}d'])
            
            # Add seasonal stratification for better model training
            day_df['Season_Indicator'] = day_df['Tanggal'].dt.month.apply(
                lambda m: 1 if m in [12, 1, 2] else  # Winter
                         2 if m in [3, 4, 5] else    # Spring
                         3 if m in [6, 7, 8] else    # Summer
                         4                           # Fall
            )
            
            # Log dataset info
            logging.info(f"Day {day} dataset shape after processing: {day_df.shape}")
            logging.info(f"Date range: {day_df['Tanggal'].min()} to {day_df['Tanggal'].max()}")
            
            # Verify no NaN values remain
            final_nan_counts = day_df.isna().sum()
            if final_nan_counts.sum() > 0:
                logging.warning(f"Warning: NaN values found in final dataset:\n{final_nan_counts}")
            
            multi_day_data[day] = day_df
        
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
        """Train separate GBM models for each forecast day"""
        results = {}
        
        # Define outlier threshold for rainfall
        RAINFALL_OUTLIER_THRESHOLD = 100  # mm
        
        # Define GBM model configuration
        base_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Process each forecast day
        for day, day_df in multi_day_data.items():
            logging.info(f"\n--- Training models for {day}-day ahead prediction ---")
            
            # Use base features for all predictions
            features_to_use = self.feature_columns
            logging.info(f"Using {len(features_to_use)} features for day {day} prediction")
            
            # Sort by date to ensure proper temporal split
            day_df = day_df.sort_values('Tanggal')
            
            # Split data while preserving temporal order
            train_size = int(0.8 * len(day_df))
            
            # Keep dates in a separate variable
            train_dates = day_df.iloc[:train_size]['Tanggal']
            test_dates = day_df.iloc[train_size:]['Tanggal']
            
            # Extract features and target
            X_train = day_df.iloc[:train_size][features_to_use]
            y_train = day_df.iloc[:train_size][f'Future_RR_{day}d']
            X_test = day_df.iloc[train_size:][features_to_use]
            y_test = day_df.iloc[train_size:][f'Future_RR_{day}d']
            
            # Log data splits info
            logging.info(f"\nTraining data: {len(X_train)} samples from {train_dates.min()} to {train_dates.max()}")
            logging.info(f"Testing data: {len(X_test)} samples from {test_dates.min()} to {test_dates.max()}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Use fewer CV folds for faster training
            cv_folds = 3 if day == 0 else 2
            
            # Initialize cross-validation
            cv_predictions = np.zeros(len(X_test))
            cv_scores = []
            
            # Show progress during training
            logging.info(f"Training GBM model with {cv_folds}-fold CV...")
            
            # Train model with cross-validation
            tscv = KFold(n_splits=cv_folds, shuffle=False)  # No shuffle for time series
            fold_predictions = []
            
            # Process each fold
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
                logging.info(f"  Processing fold {fold+1}/{cv_folds}...")
                start_time = time.time()
                
                # Split data for this fold
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
                    
                    # Handle predictions
                    # 1. Set negative values to 0
                    test_pred = np.maximum(test_pred, 0)
                    # 2. Set outlier predictions to 0
                    outlier_mask = test_pred > RAINFALL_OUTLIER_THRESHOLD
                    if np.any(outlier_mask):
                        logging.warning(f"Found {np.sum(outlier_mask)} predictions > {RAINFALL_OUTLIER_THRESHOLD}mm in fold {fold+1}")
                        test_pred[outlier_mask] = 0
                    
                    fold_predictions.append(test_pred)
                    
                    elapsed = time.time() - start_time
                    logging.info(f"    Fold R²: {r2:.4f} (took {elapsed:.1f}s)")
                    
                except Exception as e:
                    logging.error(f"Error in fold {fold+1}: {str(e)}")
                    if fold_predictions:
                        fold_predictions.append(np.mean(fold_predictions, axis=0))
                    else:
                        fold_predictions.append(np.zeros(len(X_test_scaled)))
                    cv_scores.append(0.0)
            
            # Average predictions across folds
            if fold_predictions:
                final_predictions = np.mean(fold_predictions, axis=0)
                logging.info(f"GBM CV R² scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            else:
                logging.warning(f"No valid predictions, using zeros")
                final_predictions = np.zeros(len(X_test))
            
            # Final check on predictions
            final_predictions = np.maximum(final_predictions, 0)  # Ensure no negative values
            outlier_mask = final_predictions > RAINFALL_OUTLIER_THRESHOLD
            if np.any(outlier_mask):
                logging.warning(f"Found {np.sum(outlier_mask)} final predictions > {RAINFALL_OUTLIER_THRESHOLD}mm")
                final_predictions[outlier_mask] = 0
            
            # Calculate prediction standard deviation for uncertainty estimation
            pred_std = np.std(fold_predictions, axis=0)
            
            # Calculate final metrics
            final_metrics = {
                'mse': mean_squared_error(y_test, final_predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, final_predictions)),
                'mae': mean_absolute_error(y_test, final_predictions),
                'r2': r2_score(y_test, final_predictions)
            }
            
            logging.info(f"Final metrics - R²: {final_metrics['r2']:.4f}, RMSE: {final_metrics['rmse']:.4f}")
            
            # Store results
            results[day] = {
                'test_dates': test_dates,
                'y_test': y_test,
                'y_pred': final_predictions,
                'pred_std': pred_std,
                'metrics': final_metrics,
                'features_used': features_to_use
            }
            
            # Save model data
            self.multi_day_models[day] = {
                'model': base_model,
                'scaler': scaler,
                'features_used': features_to_use,
                'pred_std': pred_std
            }
            
            # Plot results
            self._plot_prediction_with_error_bars(test_dates, y_test, final_predictions, 
                                                final_predictions - 1.96 * pred_std,
                                                final_predictions + 1.96 * pred_std,
                                                day, final_metrics)
            
        return results

    def _select_features_for_horizon(self, day, all_features):
        """Select appropriate features based on forecast horizon"""
        # Use only base features for all predictions (0 to 5 days ahead)
        return self.feature_columns

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
            
            logging.info(f"Saved GBM model for {day}-day ahead prediction in {model_dir}")

    def load_multi_day_models(self):
        """Load saved multi-day models from disk"""
        self.multi_day_models = {}
        
        # Check each day directory
        for day in range(self.forecast_days + 1):  # Include day 0
            model_dir = os.path.join(self.models_dir, f'day_{day}')
            if not os.path.exists(model_dir):
                logging.warning(f"No saved model found for day {day}")
                continue
                
            # Load model
            model_path = os.path.join(model_dir, 'gbm_model.pkl')
            if not os.path.exists(model_path):
                logging.warning(f"No model found for day {day}")
                continue
                
            model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                scaler = StandardScaler()
            
            # Load feature list
            features_path = os.path.join(model_dir, 'features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    features_used = pickle.load(f)
            else:
                features_used = self.feature_columns
            
            # Store model data
            self.multi_day_models[day] = {
                'model': model,
                'scaler': scaler,
                'features_used': features_used
            }
            
            logging.info(f"Loaded GBM model for {day}-day ahead prediction")
            
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