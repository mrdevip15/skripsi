import pandas as pd
import numpy as np
import logging
import os
import pickle
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# Create base directory for specialized rainfall model
MODEL_DIR = 'rainfall_specialist'
MODELS_DIR = os.path.join(MODEL_DIR, 'models')
PLOTS_DIR = os.path.join(MODEL_DIR, 'plots')
LOGS_DIR = os.path.join(MODEL_DIR, 'logs')
METRICS_DIR = os.path.join(MODEL_DIR, 'metrics')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'rainfall_prediction.log')),
        logging.StreamHandler()
    ]
)

class RainfallSpecialist:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        self.forecast_days = 3  # Number of days to forecast ahead
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(PLOTS_DIR, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Use domain-specific knowledge to select base features
        self.base_features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg']  # Removed ddd_x
        self.target_column = 'RR'
        
        # Track the best model for each forecast day
        self.best_models = {}
        self.feature_sets = {}
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with special focus on rainfall patterns"""
        logging.info("Starting specialized rainfall preprocessing...")
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            logging.info(f"Initial dataset size: {initial_size}")
            
            # Convert date column
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            df = df.sort_values('Tanggal')
            
            # Create binary rainfall indicators
            df['RR'] = df['RR'].replace(8888, np.nan)  # Replace missing values with NaN
            df['RainDay'] = (df['RR'] > 0).astype(int)  # Binary rain indicator
            df['LightRain'] = ((df['RR'] > 0) & (df['RR'] <= 10)).astype(int)
            df['ModerateRain'] = ((df['RR'] > 10) & (df['RR'] <= 30)).astype(int)
            df['HeavyRain'] = ((df['RR'] > 30) & (df['RR'] <= 100)).astype(int)
            df['ExtremeRain'] = (df['RR'] > 100).astype(int)
            
            # Plot rainfall distribution
            self._plot_rain_categories(df)
            
            # Remove extreme outliers (RR > 100)
            outliers = df[df['RR'] > 100].copy()
            df = df[df['RR'] <= 100].copy()
            
            # Log outlier information
            if len(outliers) > 0:
                logging.info(f"\nOutliers removed: {len(outliers)} rows")
                logging.info(f"Dates with extreme rainfall (>100mm):")
                for _, row in outliers.iterrows():
                    logging.info(f"Date: {row['Tanggal']}, Rainfall: {row['RR']}mm")
            
            # Handle missing values effectively
            # First handle categorical wind direction
            wind_dir_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            
            # Handle wind direction in ddd_car column
            if 'ddd_car' in df.columns:
                df['ddd_car_deg'] = df['ddd_car'].apply(
                    lambda x: wind_dir_map.get(str(x).strip().upper(), np.nan) if pd.notna(x) else np.nan
                )
                # Convert ddd_x to numeric, coercing errors to NaN
                df['ddd_x'] = pd.to_numeric(df['ddd_x'], errors='coerce')
                # Fill NaN values in ddd_x with values from ddd_car_deg
                df.loc[df['ddd_x'].isna(), 'ddd_x'] = df.loc[df['ddd_x'].isna(), 'ddd_car_deg']
                # Fill any remaining NaN values with the median
                df['ddd_x'] = df['ddd_x'].fillna(df['ddd_x'].median())
            
            # Handle missing values in each column appropriately
            for col in df.columns:
                if col != 'Tanggal' and df[col].isna().any():
                    # Use median for numerical columns
                    if df[col].dtype in [np.float64, np.int64]:
                        # For rainfall, use 0 for missing values as common in meteorology
                        if col == 'RR':
                            df[col] = df[col].fillna(0)
                        # For other variables, use the median
                        else:
                            df[col] = df[col].fillna(df[col].median())
            
            # Add time-based features
            df['Year'] = df['Tanggal'].dt.year
            df['Month'] = df['Tanggal'].dt.month
            df['Day'] = df['Tanggal'].dt.day
            df['DayOfYear'] = df['Tanggal'].dt.dayofyear
            df['DayOfWeek'] = df['Tanggal'].dt.dayofweek
            
            # Create comprehensive weather features
            df['Temp_Range'] = df['Tx'] - df['Tn']
            df['RH_Temp_Ratio'] = df['RH_avg'] / df['Tavg']
            df['Dew_Point'] = df['Tavg'] - ((100 - df['RH_avg']) / 5)
            
            # Create wind components
            df['Wind_Dir_Sin'] = np.sin(np.radians(df['ddd_x']))
            df['Wind_Dir_Cos'] = np.cos(np.radians(df['ddd_x']))
            df['Wind_E_Component'] = df['ff_x'] * df['Wind_Dir_Sin']
            df['Wind_N_Component'] = df['ff_x'] * df['Wind_Dir_Cos']
            
            # Add specialized rainfall features
            # Pattern of wet and dry days
            for window in [3, 7, 14, 30]:
                # Rainfall patterns
                df[f'RainFreq_{window}d'] = df['RainDay'].rolling(window=window, min_periods=1).mean()
                df[f'RR_Sum_{window}d'] = df['RR'].rolling(window=window, min_periods=1).sum()
                df[f'RR_Max_{window}d'] = df['RR'].rolling(window=window, min_periods=1).max()
                
                # Temperature and humidity patterns
                df[f'Tavg_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
                df[f'RH_{window}d'] = df['RH_avg'].rolling(window=window, min_periods=1).mean()
                df[f'Temp_Range_{window}d'] = df['Temp_Range'].rolling(window=window, min_periods=1).mean()
            
            # Add lag features with focus on most predictive lags
            key_lags = [1, 2, 3, 7]
            for lag in key_lags:
                # Previous rainfall amounts
                df[f'RR_Lag_{lag}'] = df['RR'].shift(lag)
                df[f'RainDay_Lag_{lag}'] = df['RainDay'].shift(lag)
                
                # Weather condition lags
                df[f'Tavg_Lag_{lag}'] = df['Tavg'].shift(lag)
                df[f'RH_Lag_{lag}'] = df['RH_avg'].shift(lag)
                df[f'Wind_Lag_{lag}'] = df['ff_x'].shift(lag)
            
            # Cumulative rainfall in different periods
            df['RR_Last3Days'] = df['RR_Lag_1'] + df['RR_Lag_2'] + df['RR_Lag_3']
            df['RR_LastWeek'] = df['RR_Sum_7d']
            
            # Rain transitions and patterns
            df['Rain_Yesterday'] = df['RainDay_Lag_1']
            df['Rain_2Days'] = (df['RainDay_Lag_1'] > 0) & (df['RainDay_Lag_2'] > 0)
            df['Rain_3Days'] = (df['RainDay_Lag_1'] > 0) & (df['RainDay_Lag_2'] > 0) & (df['RainDay_Lag_3'] > 0)
            
            # Seasonal features using sin/cos transformations
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12.0)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12.0)
            df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
            df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
            
            # Set date as index after creating all date-based features
            df = df.set_index('Tanggal')
            
            # Drop rows with NaN values
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Log any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                logging.info(f"\nRows dropped due to NaN values: {len(df_before_drop) - len(df)}")
            
            # Create multi-day targets
            df_multi_day = self._prepare_multi_day_dataset(df)
            
            # Create feature lists for different prediction models
            self._create_feature_sets(df)
            
            # Log final stats
            logging.info(f"\nFinal dataset size: {len(df)}")
            logging.info(f"Multi-day dataset size: {len(df_multi_day)}")
            logging.info(f"Total base features: {len(self.base_features)}")
            logging.info(f"Total enhanced features: {len(self.feature_sets['enhanced'])}")
            logging.info(f"Total rainfall pattern features: {len(self.feature_sets['rainfall_patterns'])}")
            
            # Verify no NaN values remain
            nan_counts = df.isna().sum()
            if nan_counts.sum() > 0:
                logging.warning("\nWARNING: NaN values still exist in the dataset:")
                logging.warning(nan_counts[nan_counts > 0])
            else:
                logging.info("\nNo NaN values remain in the dataset.")
            
            return df, df_multi_day
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def _plot_rain_categories(self, df):
        """Plot rainfall distribution by categories"""
        plt.figure(figsize=(14, 10))
        
        # Distribution of rainfall values
        plt.subplot(2, 2, 1)
        sns.histplot(df['RR'].dropna(), bins=20, kde=True)
        plt.title('Rainfall Distribution')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Frequency')
        
        # Plot rainfall categories
        plt.subplot(2, 2, 2)
        categories = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain']
        values = [
            sum(df['RR'] == 0),
            sum((df['RR'] > 0) & (df['RR'] <= 10)),
            sum((df['RR'] > 10) & (df['RR'] <= 30)),
            sum((df['RR'] > 30) & (df['RR'] <= 100)),
            sum(df['RR'] > 100)
        ]
        plt.bar(categories, values)
        plt.title('Rainfall Categories')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # Monthly rainfall patterns
        plt.subplot(2, 2, 3)
        monthly_rain = df.groupby(df['Tanggal'].dt.month)['RR'].mean()
        plt.plot(monthly_rain.index, monthly_rain.values, marker='o')
        plt.title('Average Monthly Rainfall')
        plt.xlabel('Month')
        plt.ylabel('Average Rainfall (mm)')
        plt.xticks(range(1, 13))
        
        # Yearly rainfall patterns
        plt.subplot(2, 2, 4)
        yearly_rain = df.groupby(df['Tanggal'].dt.year)['RR'].mean()
        plt.plot(yearly_rain.index, yearly_rain.values, marker='o')
        plt.title('Average Yearly Rainfall')
        plt.xlabel('Year')
        plt.ylabel('Average Rainfall (mm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'rainfall_categories.png'))
        plt.close()
    
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
    
    def _create_feature_sets(self, df):
        """Create different feature sets for model testing"""
        # Get all available column names
        all_cols = df.columns.tolist()
        
        # Base feature set (original weather variables)
        self.feature_sets = {
            'base': self.base_features + ['Wind_Dir_Sin', 'Wind_Dir_Cos'],  # Use processed wind direction components
            
            # Enhanced base features with best proven predictors
            'enhanced': self.base_features + [
                'Wind_Dir_Sin', 'Wind_Dir_Cos',  # Use processed wind direction components
                'RainDay', 'Temp_Range', 'RH_Temp_Ratio', 'Dew_Point',
                'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos',
                'RR_Lag_1', 'RR_Lag_2', 'RR_Lag_3',
                'Tavg_Lag_1', 'RH_Lag_1',
                'RainFreq_7d', 'RR_Sum_7d'
            ],
            
            # Rainfall-pattern specific features
            'rainfall_patterns': self.base_features + [
                'Wind_Dir_Sin', 'Wind_Dir_Cos',  # Use processed wind direction components
                'Wind_E_Component', 'Wind_N_Component',  # Add wind components
                'RainDay', 'LightRain', 'ModerateRain', 'HeavyRain',
                'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos',
                'RR_Lag_1', 'RR_Lag_2', 'RR_Lag_3', 'RR_Lag_7',
                'RainDay_Lag_1', 'RainDay_Lag_2', 'RainDay_Lag_3',
                'RainFreq_3d', 'RainFreq_7d', 'RainFreq_14d',
                'RR_Sum_3d', 'RR_Sum_7d', 'RR_Sum_14d',
                'RR_Max_3d', 'RR_Max_7d', 'RR_Max_14d',
                'RR_Last3Days', 'RR_LastWeek',
                'Rain_Yesterday', 'Rain_2Days', 'Rain_3Days'
            ],
            
            # All available features except raw wind direction
            'all': [col for col in all_cols if col not in [self.target_column, 'ddd_x', 'ddd_car', 'ddd_car_deg'] + 
                   [f'RR_Day_{day}' for day in range(1, self.forecast_days + 1)]]
        }
        
        # Log the feature sets
        for name, features in self.feature_sets.items():
            logging.info(f"\nFeature set '{name}' contains {len(features)} features")
            if len(features) <= 20:  # Only log if not too many features
                logging.info(f"Features: {', '.join(features)}")
    
    def train_and_evaluate(self, df_multi_day):
        """Train multiple models with different approaches and select best performer"""
        logging.info("Starting enhanced training and evaluation...")
        
        try:
            # Split the data chronologically
            split_idx = int(len(df_multi_day) * 0.8)
            
            # Prepare test data frame for later use
            df_test = df_multi_day.iloc[split_idx:].copy()
            
            # Train for each forecast day
            for day in range(1, self.forecast_days + 1):
                target_col = f'RR_Day_{day}'
                logging.info(f"\nTraining models for Day {day} forecast...")
                
                # Track metrics for various approaches
                day_results = {}
                
                # Try different feature sets
                for feature_set_name, features in self.feature_sets.items():
                    X = df_multi_day[features].values
                    y = df_multi_day[target_col].values
                    
                    # Apply log transformation to rainfall (adding small constant to handle zeros)
                    y_transformed = self.target_transformer.fit_transform(y.reshape(-1, 1)).ravel()
                    
                    # Split the data
                    X_train = X[:split_idx]
                    X_test = X[split_idx:]
                    y_train = y_transformed[:split_idx]
                    y_test_original = y[split_idx:]
                    
                    # Standardize features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    # Train GBM
                    params = {
                        'n_estimators': 300,
                        'learning_rate': 0.05,
                        'max_depth': 5,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'subsample': 0.8,
                        'random_state': 42
                    }
                    
                    start_time = time.time()
                    model = GradientBoostingRegressor(**params)
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions (reverse the transformation)
                    y_pred_transformed = model.predict(X_test_scaled)
                    y_pred = self.target_transformer.inverse_transform(
                        y_pred_transformed.reshape(-1, 1)).ravel()
                    
                    # Cap predictions at 0 (rainfall can't be negative)
                    y_pred = np.maximum(0, y_pred)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
                    r2 = r2_score(y_test_original, y_pred)
                    
                    # Store results
                    day_results[feature_set_name] = {
                        'model': model,
                        'features': features,
                        'rmse': rmse,
                        'r2': r2,
                        'y_pred': y_pred,
                        'scaler': self.scaler,
                        'target_transformer': self.target_transformer
                    }
                    
                    training_time = time.time() - start_time
                    
                    logging.info(f"Feature set '{feature_set_name}' results - RMSE: {rmse:.4f}, R²: {r2:.4f}, Training time: {training_time:.2f}s")
                
                # Find the best performing model based on R²
                best_set = max(day_results.items(), key=lambda x: x[1]['r2'])
                best_set_name, best_result = best_set
                
                logging.info(f"\nBest model for Day {day} uses '{best_set_name}' feature set")
                logging.info(f"Best R²: {best_result['r2']:.4f}, RMSE: {best_result['rmse']:.4f}")
                
                # Store the best model
                self.best_models[f'day_{day}'] = {
                    'model': best_result['model'],
                    'feature_set': best_set_name,
                    'features': best_result['features'],
                    'metrics': {
                        'rmse': best_result['rmse'],
                        'r2': best_result['r2']
                    },
                    'scaler': best_result['scaler'],
                    'target_transformer': best_result['target_transformer']
                }
                
                # Plot actual vs predicted for best model
                self._plot_prediction(day, y_test_original, best_result['y_pred'], df_test.index, best_set_name)
            
            # Save best models and metrics
            self._save_models_and_metrics()
            
            return self.best_models
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def _plot_prediction(self, day, y_true, y_pred, dates, model_name):
        """Plot actual vs predicted rainfall"""
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_true, label=f'Actual Day {day}', alpha=0.7)
        plt.plot(dates, y_pred, label=f'Predicted Day {day}', alpha=0.7)
        plt.title(f'Day {day} Forecast using {model_name} features')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        
        # Calculate and display metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f'RMSE: {rmse:.2f}, R²: {r2:.2f}', 
                     xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f'prediction_day_{day}_{model_name}.png'))
        plt.close()
    
    def _save_models_and_metrics(self):
        """Save best models and comprehensive metrics"""
        # Save models
        model_file = os.path.join(MODELS_DIR, f'best_models_{self.timestamp}.pkl')
        joblib.dump(self.best_models, model_file)
        logging.info(f"\nBest models saved to {model_file}")
        
        # Create and save metrics table
        metrics = {
            'Day': [],
            'Feature_Set': [],
            'Feature_Count': [],
            'RMSE': [],
            'R2': []
        }
        
        for day in range(1, self.forecast_days + 1):
            day_key = f'day_{day}'
            metrics['Day'].append(day)
            metrics['Feature_Set'].append(self.best_models[day_key]['feature_set'])
            metrics['Feature_Count'].append(len(self.best_models[day_key]['features']))
            metrics['RMSE'].append(self.best_models[day_key]['metrics']['rmse'])
            metrics['R2'].append(self.best_models[day_key]['metrics']['r2'])
        
        # Convert to DataFrame and save
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(METRICS_DIR, f'metrics_{self.timestamp}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        logging.info(f"Metrics saved to {metrics_file}")
        
        # Generate detailed report
        report_file = os.path.join(METRICS_DIR, f'report_{self.timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Rainfall Prediction Specialist Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Performance Summary:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Day':<5}{'Feature Set':<20}{'Features':<10}{'RMSE':<10}{'R²':<10}\n")
            
            for day in range(1, self.forecast_days + 1):
                day_key = f'day_{day}'
                model_info = self.best_models[day_key]
                f.write(f"{day:<5}{model_info['feature_set']:<20}{len(model_info['features']):<10}")
                f.write(f"{model_info['metrics']['rmse']:.4f}    {model_info['metrics']['r2']:.4f}\n")
            
            f.write("\nAverage R² score: {:.4f}\n".format(metrics_df['R2'].mean()))
            f.write("Average RMSE: {:.4f}\n\n".format(metrics_df['RMSE'].mean()))
            
            # List top features for Day 1 (most important day)
            day1_model = self.best_models['day_1']['model']
            day1_features = self.best_models['day_1']['features']
            
            f.write("\nTop 20 Important Features for Day 1 forecast:\n")
            f.write("-" * 50 + "\n")
            
            # Get importance scores
            importance = day1_model.feature_importances_
            indices = np.argsort(importance)[::-1][:20]  # Get top 20
            
            for i, idx in enumerate(indices):
                if idx < len(day1_features):  # Safety check
                    f.write(f"{i+1}. {day1_features[idx]} ({importance[idx]:.4f})\n")
        
        logging.info(f"Detailed report saved to {report_file}")

def main():
    try:
        # Initialize predictor
        predictor = RainfallSpecialist()
        
        # Load data
        logging.info("Loading data...")
        df = pd.read_csv('makassar.csv')
        
        # Log initial data info
        logging.info(f"Initial data shape: {df.shape}")
        
        # Preprocess data
        logging.info("\nPreprocessing data...")
        df_processed, df_multi_day = predictor.preprocess_data(df)
        
        # Train and evaluate models
        logging.info("\nTraining and evaluating models...")
        best_models = predictor.train_and_evaluate(df_multi_day)
        
        logging.info("\nRainfall prediction specialist completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 