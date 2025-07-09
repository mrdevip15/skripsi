import pandas as pd
import numpy as np
import logging
import os
import pickle
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import randint, uniform
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import time
from sklearn.feature_selection import SelectFromModel, mutual_info_regression

# Create base directory for all advanced GBM results
GBM_DIR = 'gbm_advanced'
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
        logging.FileHandler(os.path.join(LOGS_DIR, 'advanced_prediction.log')),
        logging.StreamHandler()
    ]
)

class AdvancedWeatherPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x']
        self.target_column = 'RR'
        self.scaler = StandardScaler()
        self.forecast_days = 3  # Number of days to forecast ahead
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def preprocess_data(self, df):
        """Enhanced preprocessing with outlier removal and better error handling"""
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            logging.info(f"Initial dataset size: {initial_size}")
            
            # Convert date but don't set as index yet
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            df = df.sort_values('Tanggal')
            
            # Replace 8888 with NaN in RR column
            df['RR'] = df['RR'].replace(8888, np.nan)
            
            # Plot original distribution
            self._plot_outliers(df, self.run_dir, "Original Rainfall Distribution")
            
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
            self._plot_outliers(df, self.run_dir, "Rainfall Distribution After Outlier Removal")
            
            # Binary indicator for rainfall occurrence
            df['RR_Binary'] = (df['RR'] > 0).astype(int)
            
            # Handle missing values using median imputation for better stability
            for col in df.columns:
                if col != 'Tanggal' and df[col].isna().any():
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
            
            # Add calendar features
            df['Month'] = df['Tanggal'].dt.month
            df['Day'] = df['Tanggal'].dt.day
            df['DayOfWeek'] = df['Tanggal'].dt.dayofweek
            df['DayOfYear'] = df['Tanggal'].dt.dayofyear
            df['Season'] = (df['Month'] % 12 + 3) // 3
            
            # Add weather features
            df['Temp_Range'] = df['Tx'] - df['Tn']
            df['RH_Temp_Interaction'] = df['RH_avg'] * df['Tavg']
            
            # Add derived features that meteorologists use
            
            # Dew point calculation (approximation)
            # Uses Magnus formula with Sonntag 1990 coefficients
            df['Dew_Point'] = df['Tavg'] - ((100 - df['RH_avg']) / 5)
            
            # Wet-bulb temperature estimation (simplified)
            df['Wet_Bulb_Temp'] = df['Tavg'] - 0.33 * (100 - df['RH_avg'])
            
            # Potential evapotranspiration (Hargreaves simplified)
            # Scaled for use without solar radiation data
            df['PET_Approx'] = 0.0023 * (df['Tx'] - df['Tn']).abs() ** 0.5 * (df['Tavg'] + 17.8)
            
            # Enhanced wind features - converting direction to sin/cos components
            df['Wind_Dir_Sin'] = np.sin(np.radians(df['ddd_x']))
            df['Wind_Dir_Cos'] = np.cos(np.radians(df['ddd_x']))
            
            # Wind speed weighted by direction (to capture weather patterns)
            df['Wind_E_Component'] = df['ff_x'] * df['Wind_Dir_Sin']
            df['Wind_N_Component'] = df['ff_x'] * df['Wind_Dir_Cos']
            
            # Interaction between humidity and wind
            df['RH_Wind_Interaction'] = df['RH_avg'] * df['ff_x']
            
            # Add rolling features with careful handling of NaN values
            windows = [3, 7, 14]
            for window in windows:
                # Rolling statistics for rainfall
                df[f'RR_Rolling_Mean_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).mean()
                df[f'RR_Rolling_Std_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).std()
                df[f'RR_Rolling_Max_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).max()
                
                # Frequency of rainfall
                df[f'RR_Freq_{window}d'] = df['RR_Binary'].rolling(window=window, min_periods=1).mean()
                
                # Rolling statistics for other variables
                df[f'Temp_Rolling_Mean_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
                df[f'Temp_Range_Rolling_{window}d'] = df['Temp_Range'].rolling(window=window, min_periods=1).mean()
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
                df[f'Wind_Rolling_Mean_{window}d'] = df['ff_x'].rolling(window=window, min_periods=1).mean()
            
            # Add lag features (reduced number of lags)
            for lag in [1, 2, 3, 7]:
                df[f'RR_Lag_{lag}'] = df[self.target_column].shift(lag)
                df[f'RR_Binary_Lag_{lag}'] = df['RR_Binary'].shift(lag)
                df[f'Temp_Lag_{lag}'] = df['Tavg'].shift(lag)
                df[f'RH_Lag_{lag}'] = df['RH_avg'].shift(lag)
                df[f'Wind_Lag_{lag}'] = df['ff_x'].shift(lag)
            
            # Add seasonal indicators using Fourier terms
            for period in [365.25, 30.4, 7]:  # Yearly, monthly, weekly
                for n in range(1, 3):  # Use first 2 harmonics
                    df[f'Sin_{period}_{n}'] = np.sin(2 * n * np.pi * df['DayOfYear'] / period)
                    df[f'Cos_{period}_{n}'] = np.cos(2 * n * np.pi * df['DayOfYear'] / period)
            
            # Feature interactions - especially important in weather prediction
            df['RR_Lag1_Temp'] = df['RR_Lag_1'] * df['Tavg']
            df['RR_Lag1_RH'] = df['RR_Lag_1'] * df['RH_avg']
            df['Temp_RH_Lag1'] = df['Temp_Lag_1'] * df['RH_Lag_1']
            
            # Rainfall momentum
            df['RR_1day_Change'] = df['RR'] - df['RR_Lag_1']
            df['RR_3day_Change'] = df['RR'] - df['RR_Lag_3']
            
            # Set date as index after all date-based features are created
            df = df.set_index('Tanggal')
            
            # Drop rows with NaN values that couldn't be imputed
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Log any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                logging.info(f"\nRows dropped due to NaN values: {len(df_before_drop) - len(df)}")
            
            # Create dataset for training multi-day models
            df_multi_day = self._prepare_multi_day_dataset(df)
            
            # Update feature columns with new features that were successfully created
            self.feature_columns = [
                'Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x',
                'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Season', 
                'Temp_Range', 'RH_Temp_Interaction', 'Dew_Point', 'Wet_Bulb_Temp',
                'PET_Approx', 'Wind_Dir_Sin', 'Wind_Dir_Cos', 
                'Wind_E_Component', 'Wind_N_Component', 'RH_Wind_Interaction',
                'RR_Binary'
            ]
            
            # Add rolling, lag, seasonal, and interaction features if they exist and don't have NaN values
            for col in df.columns:
                if col not in self.feature_columns and not df[col].isna().any():
                    feature_prefixes = (
                        'RR_Rolling', 'RR_Lag', 'Temp_Rolling', 'Temp_Lag', 'RH_Lag', 
                        'Wind_Rolling', 'Wind_Lag', 'RH_Rolling', 'Temp_Range_Rolling',
                        'RR_Freq', 'Sin_', 'Cos_', 'RR_Binary_Lag', 'RR_1day_Change',
                        'RR_3day_Change', 'RR_Lag1_Temp', 'RR_Lag1_RH', 'Temp_RH_Lag1'
                    )
                    if any(col.startswith(prefix) for prefix in feature_prefixes):
                        self.feature_columns.append(col)
            
            # Log final stats
            logging.info(f"\nFinal dataset size: {len(df)}")
            logging.info(f"Multi-day dataset size: {len(df_multi_day)}")
            logging.info(f"Total features: {len(self.feature_columns)}")
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
            
            return df, df_multi_day
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def _prepare_multi_day_dataset(self, df):
        """Prepare dataset for multi-day forecasting"""
        multi_day_df = df.copy()
        
        # Create target columns for each day we want to forecast
        for day in range(1, self.forecast_days + 1):
            # Shift the target backward to represent future values
            multi_day_df[f'RR_Day_{day}'] = multi_day_df[self.target_column].shift(-day)
        
        # Drop rows with NaN in target columns
        multi_day_df = multi_day_df.dropna(subset=[f'RR_Day_{day}' for day in range(1, self.forecast_days + 1)])
        
        return multi_day_df

    def _update_feature_columns(self, df):
        """Update feature columns list to include all engineered features"""
        # Start with base features
        all_features = self.feature_columns.copy()
        
        # Add time-based features
        all_features.extend(['Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range'])
        
        # Add rolling window features
        windows = [3, 7, 14, 30]
        for window in windows:
            all_features.extend([
                f'RR_Rolling_Mean_{window}d',
                f'RR_Rolling_Std_{window}d',
                f'RR_Rolling_Max_{window}d',
                f'Tavg_Rolling_Mean_{window}d',
                f'Tavg_Rolling_Std_{window}d',
                f'RH_Rolling_Mean_{window}d'
            ])
        
        # Add lag features
        max_lag = self.forecast_days + 10
        for lag in range(1, max_lag + 1):
            all_features.extend([
                f'RR_Lag_{lag}',
                f'Tavg_Lag_{lag}',
                f'RH_Lag_{lag}'
            ])
        
        # Add moving average differences
        all_features.extend([
            'RR_MA7_MA14_Diff',
            'RR_MA3_MA7_Diff'
        ])
        
        # Add cyclical encoding features
        all_features.extend([
            'Month_Sin',
            'Month_Cos',
            'DayOfWeek_Sin',
            'DayOfWeek_Cos'
        ])
        
        # Update the feature columns
        self.feature_columns = [col for col in all_features if col in df.columns]
        logging.info(f"Updated feature set with {len(self.feature_columns)} features")

    def _seasonal_decomposition(self, df):
        """Perform and plot seasonal decomposition of the rainfall data"""
        try:
            # Need reasonably continuous data for decomposition
            ts_data = df[self.target_column].asfreq('D').fillna(method='ffill')
            
            # Apply seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=365)
            
            # Plot components
            plt.figure(figsize=(12, 10))
            
            plt.subplot(411)
            decomposition.observed.plot(ax=plt.gca())
            plt.title('Observed')
            
            plt.subplot(412)
            decomposition.trend.plot(ax=plt.gca())
            plt.title('Trend')
            
            plt.subplot(413)
            decomposition.seasonal.plot(ax=plt.gca())
            plt.title('Seasonality')
            
            plt.subplot(414)
            decomposition.resid.plot(ax=plt.gca())
            plt.title('Residuals')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'seasonal_decomposition.png'))
            plt.close()
            
            logging.info("Seasonal decomposition analysis completed")
            
        except Exception as e:
            logging.warning(f"Could not perform seasonal decomposition: {str(e)}")

    def _plot_outliers(self, df, save_path, title="Rainfall Distribution"):
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

    def plot_feature_importance(self, feature_importances, feature_names, title):
        """Plot feature importance"""
        # Get top 20 features for readability
        indices = np.argsort(feature_importances)[::-1]
        top_n = min(20, len(feature_names))
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(top_n), feature_importances[indices][:top_n])
        plt.xticks(
            range(top_n), 
            [feature_names[i] for i in indices][:top_n], 
            rotation=45, 
            ha='right'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_multi_day_predictions(self, df_test, predictions_dict):
        """Plot multi-day predictions against actual values"""
        plt.figure(figsize=(15, 10))
        
        # Plot for each forecast day
        for day in range(1, self.forecast_days + 1):
            plt.subplot(self.forecast_days, 1, day)
            
            actual = df_test[f'RR_Day_{day}']
            predicted = predictions_dict[f'day_{day}']
            dates = df_test.index
            
            plt.plot(dates, actual, label=f'Actual Day {day}', alpha=0.7)
            plt.plot(dates, predicted, label=f'Predicted Day {day}', alpha=0.7)
            plt.title(f'Day {day} Forecast')
            plt.xlabel('Date')
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            
            # Calculate and display metrics
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            plt.annotate(f'RMSE: {rmse:.2f}, R²: {r2:.2f}', 
                         xy=(0.05, 0.85), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_day_predictions.png'))
        plt.close()

    def plot_prediction_intervals(self, df_test, predictions, std_devs, day=1):
        """Plot predictions with confidence intervals"""
        plt.figure(figsize=(15, 6))
        
        dates = df_test.index
        actual = df_test[f'RR_Day_{day}']
        
        # Plot actual values
        plt.plot(dates, actual, 'k-', label='Actual Values')
        
        # Plot predicted values
        plt.plot(dates, predictions, 'b-', label='Predicted Values')
        
        # Plot confidence intervals (95% = 1.96 standard deviations)
        upper_bound = predictions + 1.96 * std_devs
        lower_bound = np.maximum(0, predictions - 1.96 * std_devs)  # Rainfall can't be negative
        
        plt.fill_between(dates, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(f'Rainfall Prediction with Confidence Intervals (Day {day})')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f'prediction_intervals_day_{day}.png'))
        plt.close()

    def save_models(self, models, feature_importances, prediction_stds):
        """Save all models and metadata"""
        model_data = {
            'models': models,
            'all_feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'timestamp': self.timestamp,
            'scaler': self.scaler,
            'feature_importances': feature_importances,
            'prediction_stds': prediction_stds,
            'forecast_days': self.forecast_days
        }
        
        model_file = os.path.join(self.models_dir, f'advanced_gbm_{self.timestamp}.pkl')
        joblib.dump(model_data, model_file)
        logging.info(f"Models saved to {model_file}")

    def _save_metrics(self, metrics_dict):
        """Save detailed metrics to a CSV file"""
        metrics_dir = os.path.join(GBM_DIR, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Convert metrics dictionary to DataFrame
        metrics_df = pd.DataFrame(metrics_dict).round(4)
        
        # Save to CSV
        metrics_file = os.path.join(metrics_dir, f'metrics_{self.timestamp}.csv')
        metrics_df.to_csv(metrics_file)
        logging.info(f"Metrics saved to {metrics_file}")
        
        # Also save a detailed text report
        report_file = os.path.join(metrics_dir, f'report_{self.timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Weather Prediction Model Metrics Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write model configuration
            f.write("Model Configuration:\n")
            f.write("-" * 30 + "\n")
            f.write("GradientBoostingRegressor with parameters:\n")
            f.write("- n_estimators: 300\n")
            f.write("- learning_rate: 0.05\n")
            f.write("- max_depth: 5\n")
            f.write("- min_samples_split: 5\n")
            f.write("- min_samples_leaf: 2\n")
            f.write("- subsample: 0.8\n")
            f.write("- max_features: sqrt\n\n")
            
            for day in range(1, self.forecast_days + 1):
                f.write(f"\nDay {day} Forecast Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"RMSE: {metrics_dict['RMSE'][day-1]:.4f}\n")
                f.write(f"MAE: {metrics_dict['MAE'][day-1]:.4f}\n")
                f.write(f"R²: {metrics_dict['R2'][day-1]:.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {metrics_dict['MAPE'][day-1]:.4f}%\n")
                f.write(f"Explained Variance Score: {metrics_dict['EV'][day-1]:.4f}\n")
                f.write(f"Training Time: {metrics_dict['Training_Time'][day-1]:.2f} seconds\n")
            
            f.write("\nOverall Model Performance:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average RMSE across all days: {np.mean(metrics_dict['RMSE']):.4f}\n")
            f.write(f"Average R² across all days: {np.mean(metrics_dict['R2']):.4f}\n")
            f.write(f"Average MAPE across all days: {np.mean(metrics_dict['MAPE']):.4f}%\n")
            f.write(f"Total Training Time: {sum(metrics_dict['Training_Time']):.2f} seconds\n")
            
        logging.info(f"Detailed report saved to {report_file}")

    def _select_best_features(self, X, y, feature_names, threshold=0.01):
        """Select most important features based on correlation and mutual information"""
        # Initialize output features
        selected_features = []
        
        # Create DataFrame for analysis
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # 1. Correlation-based filtering
        # Calculate correlation with target
        correlations = {}
        for i, feature in enumerate(feature_names):
            corr = abs(np.corrcoef(X[:, i], y)[0, 1])
            if not np.isnan(corr):  # Avoid NaN correlations
                correlations[feature] = corr
        
        # Get features with correlation above threshold
        corr_selected = [f for f, c in correlations.items() if c > threshold]
        logging.info(f"Selected {len(corr_selected)} features based on correlation")
        
        # 2. Information-based filtering
        # Calculate mutual information with target
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=feature_names)
        mi_selected = mi_scores[mi_scores > mi_scores.mean()].index.tolist()
        
        logging.info(f"Selected {len(mi_selected)} features based on mutual information")
        
        # Combine selected features
        selected_features = list(set(corr_selected + mi_selected))
        logging.info(f"Combined unique selected features: {len(selected_features)}")
        
        # Always keep some basic features regardless of selection
        must_have = ['Tavg', 'RH_avg', 'RR_Lag_1', 'RR_Rolling_Mean_7d']
        for feature in must_have:
            if feature in feature_names and feature not in selected_features:
                selected_features.append(feature)
        
        logging.info(f"Final feature count after selection: {len(selected_features)}")
        logging.info(f"Selected features: {', '.join(selected_features)}")
        
        return selected_features

    def _perform_time_series_cv(self, X, y, model_params, n_splits=5):
        """Perform time series cross validation to find optimal parameters"""
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create time series split object
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Parameters to try
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        # Create RandomizedSearchCV
        search = RandomizedSearchCV(
            GradientBoostingRegressor(**model_params),
            param_grid,
            n_iter=10,  # Try 10 combinations
            cv=tscv,
            scoring='r2',
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        
        search.fit(X, y)
        
        # Log best parameters
        logging.info(f"\nBest parameters from time-series CV: {search.best_params_}")
        logging.info(f"Best R² score from CV: {search.best_score_:.4f}")
        
        return search.best_params_

    def model_tuning_and_evaluation(self, df_multi_day):
        """Train and evaluate models with simplified approach for multi-day forecasting"""
        try:
            # Prepare data - dropna as a safety measure
            df_clean = df_multi_day.dropna()
            if len(df_clean) < len(df_multi_day):
                logging.warning(f"Dropped {len(df_multi_day) - len(df_clean)} rows with NaN values before training")
            
            # Calculate split index for chronological split (80% train, 20% test)
            split_idx = int(len(df_clean) * 0.8)
            
            # Prepare test data frame for later use
            df_test = df_clean.iloc[split_idx:].copy()
            
            # Dictionary to store models and predictions for each forecast day
            models = {}
            predictions = {}
            feature_importances = {}
            prediction_stds = {}
            
            # Dictionary to store metrics
            metrics_dict = {
                'RMSE': [],
                'MAE': [],
                'R2': [],
                'MAPE': [],
                'EV': [],
                'Training_Time': []
            }
            
            # Log split information
            logging.info(f"\nTraining data shape: {df_clean.iloc[:split_idx].shape}")
            logging.info(f"Testing data shape: {df_clean.iloc[split_idx:].shape}")
            logging.info(f"Training date range: {df_clean.index[0]} to {df_clean.index[split_idx-1]}")
            logging.info(f"Testing date range: {df_clean.index[split_idx]} to {df_clean.index[-1]}")
            
            # Train a separate model for each forecast day
            for day in range(1, self.forecast_days + 1):
                logging.info(f"\nTraining model for Day {day} prediction...")
                
                # Prepare features and target
                X = df_clean[self.feature_columns].values
                target_col = f'RR_Day_{day}'
                y = df_clean[target_col].values
                
                # Split data
                X_train = X[:split_idx]
                X_test = X[split_idx:]
                y_train = y[:split_idx]
                y_test = y[split_idx:]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Use a simpler model with good defaults
                start_time = time.time()
                model = GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
                
                # Train model
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                predictions[f'day_{day}'] = y_pred
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
                ev_score = explained_variance_score(y_test, y_pred)
                
                # Store metrics
                metrics_dict['RMSE'].append(rmse)
                metrics_dict['MAE'].append(mae)
                metrics_dict['R2'].append(r2)
                metrics_dict['MAPE'].append(mape)
                metrics_dict['EV'].append(ev_score)
                metrics_dict['Training_Time'].append(training_time)
                
                # Log metrics
                logging.info(f"\nDay {day} Metrics:")
                logging.info("-" * 40)
                logging.info(f"RMSE: {rmse:.4f}")
                logging.info(f"MAE: {mae:.4f}")
                logging.info(f"R²: {r2:.4f}")
                logging.info(f"MAPE: {mape:.4f}%")
                logging.info(f"Explained Variance Score: {ev_score:.4f}")
                logging.info(f"Training Time: {training_time:.2f} seconds")
                
                # Store model and feature importance
                models[f'day_{day}'] = model
                feature_importances[f'day_{day}'] = dict(zip(self.feature_columns, model.feature_importances_))
                
                # Calculate prediction standard deviations
                residuals = y_test - y_pred
                std_dev = np.std(residuals)
                prediction_stds[f'day_{day}'] = np.ones_like(y_pred) * std_dev
                
                # Plot feature importance
                plt.figure(figsize=(12, 6))
                plt.title(f'Feature Importance for Day {day} Forecast')
                importance = pd.Series(model.feature_importances_, index=self.feature_columns)
                importance.sort_values(ascending=True).plot(kind='barh')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, f'feature_importance_day_{day}.png'))
                plt.close()
                
                # Plot prediction with confidence intervals
                self.plot_prediction_intervals(df_test, y_pred, prediction_stds[f'day_{day}'], day)
            
            # Plot multi-day predictions
            self.plot_multi_day_predictions(df_test, predictions)
            
            # Save metrics
            self._save_metrics(metrics_dict)
            
            # Save models and metadata
            self.save_models(models, feature_importances, prediction_stds)
            
            return models, predictions, df_test
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

def main():
    try:
        # Initialize predictor
        predictor = AdvancedWeatherPredictor()
        
        # Load data
        logging.info("Loading data...")
        df = pd.read_csv('makassar.csv')
        
        # Log initial data info
        logging.info(f"\nInitial data shape: {df.shape}")
        
        # Preprocess data
        logging.info("\nPreprocessing data...")
        df_processed, df_multi_day = predictor.preprocess_data(df)
        
        # Model tuning and evaluation
        logging.info("\nStarting model tuning and evaluation...")
        models, predictions, df_test = predictor.model_tuning_and_evaluation(df_multi_day)
        
        logging.info("\nAdvanced weather prediction completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 