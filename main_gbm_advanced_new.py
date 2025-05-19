import pandas as pd
import numpy as np
import logging
import os
import pickle
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import randint, uniform
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

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
        """Enhanced preprocessing with outlier removal and lag features for multi-day prediction"""
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
            
            # Plot original distribution and identify outliers
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
            
            # Now set the date as index
            df = df.set_index('Tanggal')
            
            # Handle missing values using interpolation for time series
            for col in df.columns:
                if df[col].isna().any():
                    # For RR (rainfall), use 0 for missing values as common practice
                    if col == 'RR':
                        df[col] = df[col].fillna(0)
                    else:
                        # Use interpolation for other variables
                        df[col] = df[col].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
            
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
            
            # Add time-based features
            df['Month'] = df.index.month
            df['Day'] = df.index.day
            df['DayOfWeek'] = df.index.dayofweek
            df['Season'] = (df.index.month % 12 + 3) // 3
            df['Temp_Range'] = df['Tx'] - df['Tn']
            
            # Create features for multi-day prediction
            
            # 1. Add more comprehensive rolling windows
            windows = [3, 7, 14, 30]
            for window in windows:
                # Rolling statistics
                df[f'RR_Rolling_Mean_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).mean()
                df[f'RR_Rolling_Std_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).std()
                df[f'RR_Rolling_Max_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).max()
                
                df[f'Tavg_Rolling_Mean_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
                df[f'Tavg_Rolling_Std_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).std()
                
                # Also add rolling features for humidity
                df[f'RH_Rolling_Mean_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
            
            # 2. Add lag features for each target forecast day
            # Create lags up to forecast_days + buffer
            max_lag = self.forecast_days + 10  # Additional lags as buffer
            for lag in range(1, max_lag + 1):
                df[f'RR_Lag_{lag}'] = df[self.target_column].shift(lag)
                df[f'Tavg_Lag_{lag}'] = df['Tavg'].shift(lag)
                df[f'RH_Lag_{lag}'] = df['RH_avg'].shift(lag)
            
            # 3. Add moving average lag differences (momentum indicators)
            df['RR_MA7_MA14_Diff'] = df['RR_Rolling_Mean_7d'] - df['RR_Rolling_Mean_14d']
            df['RR_MA3_MA7_Diff'] = df['RR_Rolling_Mean_3d'] - df['RR_Rolling_Mean_7d']
            
            # 4. Add cyclical encoding for month and day of week for better seasonality capture
            df['Month_Sin'] = np.sin(2 * np.pi * df.index.month/12)
            df['Month_Cos'] = np.cos(2 * np.pi * df.index.month/12)
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
            
            # Drop rows with NaN values that couldn't be imputed
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Log any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                logging.info(f"\nRows dropped due to NaN values: {len(df_before_drop) - len(df)}")
            
            # Create dataset for training multi-day models
            df_multi_day = self._prepare_multi_day_dataset(df)
            
            # Update feature columns with created features
            self._update_feature_columns(df)
            
            # Final dataset stats
            logging.info(f"\nFinal dataset size: {len(df)}")
            logging.info(f"Multi-day dataset size: {len(df_multi_day)}")
            logging.info(f"Total features: {len(self.feature_columns)}")
            
            # Create seasonal decomposition and plot
            self._seasonal_decomposition(df)
            
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

    def model_tuning_and_evaluation(self, df_multi_day):
        """Train and evaluate models with hyperparameter tuning for multi-day forecasting"""
        try:
            # Prepare data
            X = df_multi_day[self.feature_columns].values
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Dictionary to store models and predictions for each forecast day
            models = {}
            predictions = {}
            feature_importances = {}
            prediction_stds = {}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Calculate split index for final evaluation
            split_idx = int(len(X_scaled) * 0.8)
            
            # Split data
            X_train = X_scaled[:split_idx]
            X_test = X_scaled[split_idx:]
            df_test = df_multi_day.iloc[split_idx:].copy()
            
            # Parameter grid for RandomizedSearchCV
            param_dist = {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.19),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Train a separate model for each forecast day
            for day in range(1, self.forecast_days + 1):
                target_col = f'RR_Day_{day}'
                y = df_multi_day[target_col].values
                
                # Split target
                y_train = y[:split_idx]
                y_test = y[split_idx:]
                
                logging.info(f"\nTraining model for Day {day} prediction...")
                
                # Initialize base model
                base_model = GradientBoostingRegressor(random_state=42)
                
                # Randomized search with time series cross-validation
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=20,  # Number of parameter settings sampled
                    cv=tscv,
                    scoring='neg_root_mean_squared_error',
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit the model
                random_search.fit(X_train, y_train)
                
                # Get best model
                best_model = random_search.best_estimator_
                models[f'day_{day}'] = best_model
                
                # Log best parameters
                logging.info(f"Best parameters for Day {day} model:")
                logging.info(random_search.best_params_)
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                predictions[f'day_{day}'] = y_pred
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"Day {day} Metrics:")
                logging.info(f"RMSE: {rmse:.4f}")
                logging.info(f"MAE: {mae:.4f}")
                logging.info(f"R²: {r2:.4f}")
                
                # Calculate prediction standard deviations (for confidence intervals)
                # Using prediction error on test set as a proxy
                residuals = y_test - y_pred
                std_dev = np.std(residuals)
                prediction_stds[f'day_{day}'] = np.ones_like(y_pred) * std_dev
                
                # Get feature importance
                feature_importances[f'day_{day}'] = best_model.feature_importances_
                
                # Plot feature importance
                self.plot_feature_importance(
                    best_model.feature_importances_,
                    self.feature_columns,
                    f'Feature Importance Day {day}'
                )
                
                # For Day 1, also calculate permutation importance for more robust feature importance
                if day == 1:
                    perm_importance = permutation_importance(
                        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
                    )
                    
                    self.plot_feature_importance(
                        perm_importance.importances_mean,
                        self.feature_columns,
                        'Permutation Feature Importance Day 1'
                    )
                
                # Plot prediction with confidence intervals
                self.plot_prediction_intervals(df_test, y_pred, prediction_stds[f'day_{day}'], day)
            
            # Plot multi-day predictions
            self.plot_multi_day_predictions(df_test, predictions)
            
            # Save all models and metadata
            self.save_models(models, feature_importances, prediction_stds)
            
            return models, predictions, df_test
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    def save_models(self, models, feature_importances, prediction_stds):
        """Save all models and metadata"""
        model_data = {
            'models': models,
            'feature_columns': self.feature_columns,
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