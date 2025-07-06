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
GBM_DIR = 'gbm_imakas'
MODELS_DIR = os.path.join(GBM_DIR, 'models')
PLOTS_DIR = os.path.join(GBM_DIR, 'plots')
LOGS_DIR = os.path.join(GBM_DIR, 'logs')

# Create necessary directories
os.makedirs(GBM_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class IMakasWeatherPredictor:
    def __init__(self):
        self.model = None
        self.multi_day_models = {}  # For storing models for different forecast horizons
        self.multi_target_models = {}  # For storing models for different target variables
        
        # Define target variables based on IMAKAS7.csv structure
        self.target_columns = ['Humidity_%', 'Temperature_C', 'Precip_Rate_mm', 'Precip_Accum_mm', 'Solar_w/m2']
        self.target_names = {
            'Humidity_%': 'Kelembaban (%)',
            'Temperature_C': 'Suhu (°C)',
            'Precip_Rate_mm': 'Laju Curah Hujan (mm)',
            'Precip_Accum_mm': 'Akumulasi Curah Hujan (mm)',
            'Solar_w/m2': 'Radiasi Matahari (W/m²)'
        }
        
        # Define feature columns (excluding target variables)
        self.feature_columns = [
            # Basic weather measurements (excluding targets)
            'Dew_Point_C', 'Speed_kmh', 'Gust_kmh', 'Pressure_hPa', 'UV',
            
            # Time-based features (will be created during preprocessing)
            'Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'Season',
            'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
            
            # Derived features
            'Temp_Dew_Diff', 'Wind_Pressure_Ratio', 'Vapor_Pressure_Deficit',
            
            # Rolling features (will be created)
            'Temperature_C_Rolling_Mean_6h', 'Temperature_C_Rolling_Std_6h',
            'Humidity_%_Rolling_Mean_6h', 'Humidity_%_Rolling_Std_6h',
            'Pressure_hPa_Rolling_Mean_6h', 'Pressure_hPa_Rolling_Std_6h',
            
            # Lag features (will be created)
            'Temperature_C_Lag_1h', 'Temperature_C_Lag_3h', 'Temperature_C_Lag_6h',
            'Humidity_%_Lag_1h', 'Humidity_%_Lag_3h', 'Humidity_%_Lag_6h',
            'Pressure_hPa_Lag_1h', 'Pressure_hPa_Lag_3h',
            
            # Weather pattern indicators
            'Rain_Event', 'Solar_Active', 'High_Humidity_Event'
        ]
        
        self.scaler = StandardScaler()
        self.target_scalers = {}  # Separate scalers for each target
        self.forecast_hours = [1, 3, 6, 12, 24]  # Hours to forecast ahead
        self.cv_folds = 5  # Number of cross-validation folds
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def plot_target_distributions(self, df, save_path, title_prefix="Distribusi Variabel Target"):
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
                plt.ylabel('Frekuensi')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{title_prefix.lower().replace(' ', '_')}.png"))
        plt.close()

    def preprocess_data(self, df):
        """Enhanced preprocessing for IMAKAS7.csv data"""
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            print(f"Initial dataset size: {initial_size}")
            
            # Convert date and time to datetime
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.sort_values('DateTime')
            
            # Extract time-based features
            df['Hour'] = df['DateTime'].dt.hour
            df['Day'] = df['DateTime'].dt.day
            df['Month'] = df['DateTime'].dt.month
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['DayOfYear'] = df['DateTime'].dt.dayofyear
            df['Season'] = (df['Month'] % 12 + 3) // 3
            
            # Add cyclical encoding of time features
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
            df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
            df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
            
            # Handle missing values and clean data
            # Remove rows with missing essential data
            essential_cols = ['Temperature_C', 'Humidity_%', 'Pressure_hPa', 'Dew_Point_C']
            df = df.dropna(subset=essential_cols)
            
            # Fill missing values in other columns
            numeric_cols = ['Speed_kmh', 'Gust_kmh', 'Precip_Rate_mm', 'Precip_Accum_mm', 'UV', 'Solar_w/m2']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)  # Assume 0 for missing precipitation, UV, solar
            
            # Create derived features
            df['Temp_Dew_Diff'] = df['Temperature_C'] - df['Dew_Point_C']
            df['Wind_Pressure_Ratio'] = df['Speed_kmh'] / (df['Pressure_hPa'] / 1000)
            
            # Vapor Pressure Deficit (important for humidity prediction)
            # Simplified VPD calculation
            df['Vapor_Pressure_Deficit'] = df['Temp_Dew_Diff'] * 0.1  # Simplified approximation
            
            # Create weather event indicators
            df['Rain_Event'] = (df['Precip_Rate_mm'] > 0).astype(int)
            df['Solar_Active'] = (df['Solar_w/m2'] > 100).astype(int)
            df['High_Humidity_Event'] = (df['Humidity_%'] > 80).astype(int)
            
            # Create rolling features (6-hour windows)
            rolling_window = 12  # 12 records = ~6 hours (assuming 30-min intervals)
            for target in self.target_columns:
                if target in df.columns:
                    df[f'{target}_Rolling_Mean_6h'] = df[target].rolling(window=rolling_window, min_periods=1).mean()
                    df[f'{target}_Rolling_Std_6h'] = df[target].rolling(window=rolling_window, min_periods=1).std()
            
            # Add rolling features for key variables
            df['Pressure_hPa_Rolling_Mean_6h'] = df['Pressure_hPa'].rolling(window=rolling_window, min_periods=1).mean()
            df['Pressure_hPa_Rolling_Std_6h'] = df['Pressure_hPa'].rolling(window=rolling_window, min_periods=1).std()
            
            # Create lag features (1h, 3h, 6h lags)
            lag_steps = [2, 6, 12]  # Assuming ~30-min intervals: 2=1h, 6=3h, 12=6h
            lag_names = ['1h', '3h', '6h']
            
            for target in self.target_columns:
                if target in df.columns:
                    for lag_step, lag_name in zip(lag_steps, lag_names):
                        df[f'{target}_Lag_{lag_name}'] = df[target].shift(lag_step)
            
            # Add lag features for pressure
            for lag_step, lag_name in zip(lag_steps[:2], lag_names[:2]):  # Only 1h and 3h lags for pressure
                df[f'Pressure_hPa_Lag_{lag_name}'] = df['Pressure_hPa'].shift(lag_step)
            
            # Handle NaN values created by rolling and lag features
            # Forward fill for lag features
            lag_cols = [col for col in df.columns if '_Lag_' in col]
            for col in lag_cols:
                df[col] = df[col].fillna(method='ffill')
            
            # Forward fill for rolling features
            rolling_cols = [col for col in df.columns if '_Rolling_' in col]
            for col in rolling_cols:
                df[col] = df[col].fillna(method='ffill')
            
            # Fill any remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining rows with NaN values
            df = df.dropna()
            
            # Update feature columns list to only include existing columns
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]
            
            # Plot original distribution
            self.plot_target_distributions(df, self.run_dir, "Distribusi Awal Variabel Target")
            
            # Print final stats
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
            
            print(f"\nFinal feature list ({len(self.feature_columns)} features):")
            print(', '.join(self.feature_columns))
            
            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def plot_feature_correlations(self, df):
        """Plot correlation matrix of features with all target variables"""
        # Use only a subset of features if there are too many
        if len(self.feature_columns) > 15:
            # Calculate correlation with all targets
            feature_importance = {}
            for feature in self.feature_columns:
                if feature in df.columns:
                    correlations = []
                    for target in self.target_columns:
                        if target in df.columns:
                            corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                            if not np.isnan(corr):
                                correlations.append(corr)
                    feature_importance[feature] = np.mean(correlations) if correlations else 0
            
            # Get the top 15 most correlated features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_features = [f[0] for f in top_features]
            features_to_plot = selected_features + self.target_columns
            print(f"\nPlotting correlations for top 15 features: {', '.join(selected_features)}")
        else:
            features_to_plot = self.feature_columns + self.target_columns
        
        # Filter features that exist in the dataframe
        features_to_plot = [f for f in features_to_plot if f in df.columns]
        
        # Create correlation matrix
        corr = df[features_to_plot].corr()
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Korelasi Fitur dengan Variabel Target')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_correlations.png'))
        plt.close()

    def plot_seasonal_patterns(self, df):
        """Plot hourly and monthly patterns of all target variables"""
        n_targets = len([t for t in self.target_columns if t in df.columns])
        
        # Hourly patterns
        plt.figure(figsize=(15, 4*n_targets))
        plot_idx = 1
        
        for target in self.target_columns:
            if target in df.columns:
                plt.subplot(n_targets, 2, plot_idx)
                hourly_avg = df.groupby('Hour')[target].mean()
                hourly_avg.plot(kind='line', marker='o')
                plt.title(f'Rata-rata {self.target_names[target]} per Jam')
                plt.xlabel('Jam')
                plt.ylabel(self.target_names[target])
                plt.grid(True, alpha=0.3)
                
                plt.subplot(n_targets, 2, plot_idx + 1)
                monthly_avg = df.groupby('Month')[target].mean()
                monthly_avg.plot(kind='bar')
                plt.title(f'Rata-rata {self.target_names[target]} per Bulan')
                plt.xlabel('Bulan')
                plt.ylabel(self.target_names[target])
                plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                                     'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'], rotation=45)
                
                plot_idx += 2
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'temporal_patterns.png'))
        plt.close()

    def plot_predictions(self, dates, actual, predicted, target_name='Target'):
        """Plot actual vs predicted values for a specific target"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, actual, marker='o', linestyle='-', label='Aktual', alpha=0.7, markersize=2)
        plt.plot(dates, predicted, marker='x', linestyle='-', label='Prediksi', alpha=0.7, markersize=2)
        plt.title(f'Aktual vs Prediksi {target_name}')
        plt.xlabel('Tanggal')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Improved filename sanitization
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
            
            # Sort features by importance (show top 8)
            indices = np.argsort(feature_importance)[::-1][:8]
            sorted_feature_names = [self.feature_columns[idx] for idx in indices]
            sorted_importance = feature_importance[indices]
            
            # Plot horizontal bar chart
            plt.barh(range(len(sorted_importance)), sorted_importance)
            plt.yticks(range(len(sorted_importance)), sorted_feature_names)
            plt.xlabel('Kepentingan')
            plt.title(f'8 Fitur Teratas - {self.target_names[target]}')
            plt.gca().invert_yaxis()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_target_feature_importance.png'))
        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        if not self.multi_target_models:
            print("Models not trained yet, cannot plot feature importance.")
            return
            
        self.plot_multi_target_feature_importance(self.multi_target_models)

    def save_metrics(self, metrics, target_name=''):
        """Save evaluation metrics to a file"""
        if target_name:
            # Sanitize filename to remove problematic characters
            safe_target_name = (target_name.replace('/', '_')
                               .replace('\\', '_')
                               .replace(':', '_')
                               .replace('*', '_')
                               .replace('?', '_')
                               .replace('"', '_')
                               .replace('<', '_')
                               .replace('>', '_')
                               .replace('|', '_')
                               .replace('%', 'pct'))
            filename = f'metrics_{safe_target_name}.txt'
        else:
            filename = 'metrics.txt'
        with open(os.path.join(self.run_dir, filename), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def prepare_multi_hour_dataset(self, df):
        """Prepare dataset for multi-hour forecasting"""
        multi_hour_data = {}
        
        df_copy = df.copy()
        
        # Ensure DateTime is the index or a column
        if 'DateTime' not in df_copy.columns:
            df_copy.reset_index(inplace=True)
        
        # Sort data by datetime
        df_copy = df_copy.sort_values('DateTime')
        
        print(f"\nInitial data shape: {df_copy.shape}")
        
        # For each target variable, create datasets for each forecast hour
        for target in self.target_columns:
            if target not in df_copy.columns:
                continue
                
            multi_hour_data[target] = {}
            
            # For each forecast hour
            for hours in self.forecast_hours:
                print(f"\nPreparing {target} dataset for {hours}-hour ahead prediction")
                
                # Create dataset with base features
                hour_df = df_copy[['DateTime'] + self.feature_columns].copy()
                
                # Calculate the number of steps for the forecast
                # Assuming data is collected every 30 minutes on average
                steps = hours * 2  # 2 records per hour
                
                # Shift target variable to create future target
                hour_df[f'Future_{target}_{hours}h'] = df_copy[target].shift(-steps)
                
                # Drop rows with NaN in target
                hour_df = hour_df.dropna(subset=[f'Future_{target}_{hours}h'])
                
                # Log dataset info
                print(f"  {target} {hours}h dataset shape: {hour_df.shape}")
                print(f"  DateTime range: {hour_df['DateTime'].min()} to {hour_df['DateTime'].max()}")
                
                multi_hour_data[target][hours] = hour_df
        
        return multi_hour_data

    def train_multi_hour_models(self, multi_hour_data):
        """Train separate GBM models for each target and forecast hour"""
        results = {}
        
        # Define base model configurations for different targets
        model_configs = {
            'Humidity_%': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                min_samples_split=10, min_samples_leaf=5, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'Temperature_C': GradientBoostingRegressor(
                n_estimators=180, learning_rate=0.06, max_depth=5,
                min_samples_split=8, min_samples_leaf=4, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'Precip_Rate_mm': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'Precip_Accum_mm': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                max_features='sqrt', random_state=42
            ),
            'Solar_w/m2': GradientBoostingRegressor(
                n_estimators=160, learning_rate=0.07, max_depth=5,
                min_samples_split=6, min_samples_leaf=4, subsample=0.8,
                max_features='sqrt', random_state=42
            )
        }
        
        # Process each target variable
        for target in self.target_columns:
            if target not in multi_hour_data:
                print(f"Skipping {target} - no multi-hour data available")
                continue
                
            print(f"\n=== Training Multi-Hour Models for {self.target_names[target]} ===")
            results[target] = {}
            
            # Process each forecast hour for this target
            for hours, hour_df in multi_hour_data[target].items():
                print(f"\n--- {target}: {hours}-hour ahead prediction ---")
                
                # Use base features for all predictions
                features_to_use = self.feature_columns
                print(f"Using {len(features_to_use)} features for {target} {hours}h prediction")
                
                # Sort by datetime
                hour_df = hour_df.sort_values('DateTime')
                
                # Split data while preserving temporal order
                train_size = int(0.8 * len(hour_df))
                
                # Keep dates in a separate variable
                train_dates = hour_df.iloc[:train_size]['DateTime']
                test_dates = hour_df.iloc[train_size:]['DateTime']
                
                # Extract features and target
                X_train = hour_df.iloc[:train_size][features_to_use]
                y_train = hour_df.iloc[:train_size][f'Future_{target}_{hours}h']
                X_test = hour_df.iloc[train_size:][features_to_use]
                y_test = hour_df.iloc[train_size:][f'Future_{target}_{hours}h']
                
                # Log data splits info
                print(f"Training data: {len(X_train)} samples from {train_dates.min()} to {train_dates.max()}")
                print(f"Testing data: {len(X_test)} samples from {test_dates.min()} to {test_dates.max()}")
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use fewer CV folds for faster training
                cv_folds = 3
                
                # Get base model for this target
                base_model = model_configs.get(target, model_configs['Temperature_C'])
                
                # Train model with cross-validation
                tscv = KFold(n_splits=cv_folds, shuffle=False)
                fold_predictions = []
                cv_scores = []
                
                print(f"Training {target} GBM model with {cv_folds}-fold CV...")
                
                # Process each fold
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
                    print(f"  Processing fold {fold+1}/{cv_folds}...")
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
                        
                        # Apply target-specific constraints
                        if target == 'Humidity_%':
                            test_pred = np.clip(test_pred, 0, 100)
                        elif target in ['Precip_Rate_mm', 'Precip_Accum_mm']:
                            test_pred = np.maximum(test_pred, 0)
                        elif target == 'Solar_w/m2':
                            test_pred = np.maximum(test_pred, 0)
                        
                        fold_predictions.append(test_pred)
                        
                        elapsed = time.time() - start_time
                        print(f"    Fold R²: {r2:.4f} (took {elapsed:.1f}s)")
                        
                    except Exception as e:
                        print(f"Error in fold {fold+1}: {str(e)}")
                        if fold_predictions:
                            fold_predictions.append(np.mean(fold_predictions, axis=0))
                        else:
                            fold_predictions.append(np.zeros(len(X_test_scaled)))
                        cv_scores.append(0.0)
                
                # Average predictions across folds
                if fold_predictions:
                    final_predictions = np.mean(fold_predictions, axis=0)
                    print(f"{target} GBM CV R² scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                else:
                    print(f"No valid predictions for {target}, using zeros")
                    final_predictions = np.zeros(len(X_test))
                
                # Apply final constraints
                if target == 'Humidity_%':
                    final_predictions = np.clip(final_predictions, 0, 100)
                elif target in ['Precip_Rate_mm', 'Precip_Accum_mm']:
                    final_predictions = np.maximum(final_predictions, 0)
                elif target == 'Solar_w/m2':
                    final_predictions = np.maximum(final_predictions, 0)
                
                # Calculate final metrics
                final_metrics = {
                    'mse': mean_squared_error(y_test, final_predictions),
                    'rmse': np.sqrt(mean_squared_error(y_test, final_predictions)),
                    'mae': mean_absolute_error(y_test, final_predictions),
                    'r2': r2_score(y_test, final_predictions)
                }
                
                print(f"Final {target} metrics - R²: {final_metrics['r2']:.4f}, RMSE: {final_metrics['rmse']:.4f}")
                
                # Store results
                results[target][hours] = {
                    'test_dates': test_dates,
                    'y_test': y_test,
                    'y_pred': final_predictions,
                    'metrics': final_metrics,
                    'features_used': features_to_use
                }
                
                # Plot results for this target and hour
                hour_label = f"{hours} Jam Ke Depan"
                target_name = self.target_names[target]
                self.plot_predictions(test_dates, y_test, final_predictions, f"{target_name} - {hour_label}")
                
        return results

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
                
                # Scale target if needed
                target_scaler = StandardScaler()
                if target in ['Temperature_C', 'Precip_Rate_mm', 'Solar_w/m2']:
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    self.target_scalers[target] = target_scaler
                else:
                    y_train_scaled = y_train.values
                    self.target_scalers[target] = None
                
                # Define model for this target
                if target == 'Humidity_%':
                    model = GradientBoostingRegressor(
                        n_estimators=150, learning_rate=0.08, max_depth=4,
                        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
                        random_state=42
                    )
                elif target == 'Temperature_C':
                    model = GradientBoostingRegressor(
                        n_estimators=180, learning_rate=0.06, max_depth=5,
                        min_samples_split=8, min_samples_leaf=4, subsample=0.8,
                        random_state=42
                    )
                elif target in ['Precip_Rate_mm', 'Precip_Accum_mm']:
                    model = GradientBoostingRegressor(
                        n_estimators=200, learning_rate=0.05, max_depth=6,
                        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
                        random_state=42
                    )
                else:  # Solar_w/m2
                    model = GradientBoostingRegressor(
                        n_estimators=160, learning_rate=0.07, max_depth=5,
                        min_samples_split=6, min_samples_leaf=4, subsample=0.8,
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
                if target == 'Humidity_%':
                    y_pred = np.clip(y_pred, 0, 100)
                elif target in ['Precip_Rate_mm', 'Precip_Accum_mm']:
                    y_pred = np.maximum(y_pred, 0)
                elif target == 'Solar_w/m2':
                    y_pred = np.maximum(y_pred, 0)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
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
                test_dates = df.iloc[train_size:]['DateTime'] if 'DateTime' in df.columns else range(len(y_test))
                self.plot_predictions(test_dates, y_test, y_pred, self.target_names[target])
                
                # Store for summary
                all_metrics[target] = self.multi_target_models[target]['metrics']
            
            # Print summary of all models
            print(f"\n=== Multi-Target Model Summary ===")
            for target, metrics in all_metrics.items():
                print(f"{self.target_names[target]:30} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Plot comprehensive visualizations
            print(f"\nGenerating visualization plots...")
            self.plot_feature_importance()
            self.plot_feature_correlations(df)
            self.plot_seasonal_patterns(df)
            
            # Multi-hour forecasting for all targets
            print(f"\n=== Multi-Hour Forecasting ===")
            print("Preparing multi-hour forecasting dataset...")
            multi_hour_data = self.prepare_multi_hour_dataset(df)
            
            print("Training multi-hour forecasting models...")
            multi_hour_results = self.train_multi_hour_models(multi_hour_data)
            
            return all_metrics
            
        except Exception as e:
            print(f"Error in model training and evaluation: {str(e)}")
            raise

def main():
    try:
        # Load dataset
        print("Loading IMAKAS7 dataset...")
        df = pd.read_csv('IMAKAS7.csv')
        
        # Initialize predictor
        predictor = IMakasWeatherPredictor()
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Train and evaluate
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