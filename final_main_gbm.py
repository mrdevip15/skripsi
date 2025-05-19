import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.inspection import permutation_importance
import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

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
        
        # Log state after creating historical features
        logging.info(f"\nAfter adding historical features - shape: {df_copy.shape}")
        logging.info(f"NaN counts after historical features:\n{df_copy.isna().sum()}")
        
        # Update feature columns to include historical features
        historical_features = []
        for feature in self.feature_columns:
            historical_features.extend([f'{feature}_past_{i}d' for i in range(1, 6)])
        
        # Combine current and historical features
        self.extended_features = self.feature_columns + historical_features
        
        # Handle missing values in historical features using forward fill and backward fill
        for feature in self.extended_features:
            # First try forward fill
            df_copy[feature] = df_copy[feature].fillna(method='ffill')
            # Then backward fill any remaining NaNs
            df_copy[feature] = df_copy[feature].fillna(method='bfill')
            # Finally, if any NaNs remain, fill with column median
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
            
            # Log info about the dataset for this day
            logging.info(f"\nDay {day} dataset - shape before dropping NaN: {len(day_df)}")
            logging.info(f"Number of NaN values in day {day} dataset: {day_df.isna().sum().sum()}")
            
            multi_day_data[day] = day_df
            
            # Log final dataset size for this day
            logging.info(f"Final size of day {day} dataset: {len(multi_day_data[day])}")
        
        return multi_day_data

    def plot_feature_importance_extended(self, feature_importance, day):
        """Plot feature importance including historical features"""
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [self.extended_features[i] for i in indices]
        sorted_importance = feature_importance[indices]
        
        # Group features by type (current vs historical)
        current_features = []
        historical_features = []
        importances_current = []
        importances_historical = []
        
        for name, importance in zip(sorted_feature_names, sorted_importance):
            if '_past_' in name:
                historical_features.append(name)
                importances_historical.append(importance)
            else:
                current_features.append(name)
                importances_current.append(importance)
        
        # Plot separate graphs for current and historical features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Current features
        y_pos = np.arange(len(current_features))
        ax1.barh(y_pos, importances_current)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(current_features)
        ax1.set_title('Current Day Features Importance')
        
        # Historical features
        y_pos = np.arange(len(historical_features))
        ax2.barh(y_pos, importances_historical)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(historical_features)
        ax2.set_title('Historical Features Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f'feature_importance_day{day}.png'))
        plt.close()

    def train_multi_day_models(self, multi_day_data):
        """Train separate models for each forecast day using ensemble approach"""
        results = {}
        
        # Model types to try for each day
        model_types = {
            'gbm': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50), 
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        }
        
        for day, day_df in multi_day_data.items():
            logging.info(f"\n--- Training models for {day}-day ahead prediction ---")
            
            # Split data
            train_size = int(0.8 * len(day_df))
            
            # Keep dates in a separate variable
            train_dates = day_df.iloc[:train_size]['Tanggal']
            test_dates = day_df.iloc[train_size:]['Tanggal']
            
            # Extract features and target without the date column
            X_train = day_df.iloc[:train_size].drop(['Tanggal', f'Future_RR_{day}d'], axis=1)
            y_train = day_df.iloc[:train_size][f'Future_RR_{day}d']
            X_test = day_df.iloc[train_size:].drop(['Tanggal', f'Future_RR_{day}d'], axis=1)
            y_test = day_df.iloc[train_size:][f'Future_RR_{day}d']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate each model type
            model_results = {}
            best_r2 = -float('inf')
            best_model_name = None
            best_model = None
            all_predictions = []
            
            for model_name, model in model_types.items():
                logging.info(f"Training {model_name} model for day {day}...")
                
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    all_predictions.append(y_pred)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    logging.info(f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                    
                    # Save model results
                    model_results[model_name] = {
                        'model': model,
                        'metrics': {
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2
                        },
                        'predictions': y_pred
                    }
                    
                    # Track best model
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = model_name
                        best_model = model
                
                except Exception as e:
                    logging.error(f"Error training {model_name} model: {str(e)}")
            
            # Create ensemble prediction (simple average of all models)
            if all_predictions:
                ensemble_pred = np.mean(all_predictions, axis=0)
                ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                ensemble_rmse = np.sqrt(ensemble_mse)
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                
                logging.info(f"Ensemble - MSE: {ensemble_mse:.4f}, RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
                
                model_results['ensemble'] = {
                    'model': 'ensemble',
                    'metrics': {
                        'mse': ensemble_mse,
                        'rmse': ensemble_rmse,
                        'mae': ensemble_mae,
                        'r2': ensemble_r2
                    },
                    'predictions': ensemble_pred
                }
                
                # Check if ensemble is better than individual models
                if ensemble_r2 > best_r2:
                    best_r2 = ensemble_r2
                    best_model_name = 'ensemble'
            
            logging.info(f"Best model for day {day}: {best_model_name} with R²: {best_r2:.4f}")
            
            # Use best model or ensemble for final predictions
            if best_model_name == 'ensemble':
                final_pred = ensemble_pred
                metrics = model_results['ensemble']['metrics']
                # For ensemble, we use the first model for importance analysis
                if 'gbm' in model_results:
                    feature_importance = model_results['gbm']['model'].feature_importances_
                else:
                    # If GBM not available, use dummy feature importance
                    feature_importance = np.ones(len(self.extended_features)) / len(self.extended_features)
            else:
                final_pred = model_results[best_model_name]['predictions']
                metrics = model_results[best_model_name]['metrics']
                # Get feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = best_model.feature_importances_
                else:
                    # For models without direct feature importance, use permutation importance
                    perm_imp = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
                    feature_importance = perm_imp.importances_mean
            
            # Plot feature importance
            self.plot_feature_importance_extended(feature_importance, day)
            
            # Log top important features
            feature_importance_list = list(zip(self.extended_features, feature_importance))
            feature_importance_list.sort(key=lambda x: x[1], reverse=True)
            logging.info(f"\nTop 10 most important features for {day}-day ahead prediction:")
            for feature, importance in feature_importance_list[:10]:
                logging.info(f"{feature}: {importance:.4f}")
            
            # Estimate prediction intervals
            pred_std = np.std(y_test - final_pred)
            
            # Store final results
            results[day] = {
                'model': best_model_name,
                'scaler': scaler,
                'metrics': metrics,
                'test_dates': test_dates,
                'y_test': y_test,
                'y_pred': final_pred,
                'pred_std': pred_std,
                'feature_importance': feature_importance
            }
            
            # Save the best model and scaler
            self.multi_day_models[day] = {
                'model_name': best_model_name,
                'model': best_model,
                'scaler': scaler,
                'pred_std': pred_std,
                'feature_importance': feature_importance
            }
        
        return results

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
            model_path = os.path.join(self.models_dir, f'gbm_model_day{day}.pkl')
            scaler_path = os.path.join(self.models_dir, f'scaler_day{day}.pkl')
            imp_path = os.path.join(self.models_dir, f'feature_importance_day{day}.pkl')
            std_path = os.path.join(self.models_dir, f'pred_std_day{day}.pkl')
            
            # Save model and scaler
            joblib.dump(model_data['model'], model_path)
            joblib.dump(model_data['scaler'], scaler_path)
            
            # Save feature importance and prediction std
            with open(imp_path, 'wb') as f:
                pickle.dump(model_data['feature_importance'], f)
                
            with open(std_path, 'wb') as f:
                pickle.dump(model_data['pred_std'], f)
            
            logging.info(f"Saved model for {day}-day ahead prediction")

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
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 