import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
        self.feature_columns = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x']
        self.target_column = 'RR'
        self.scaler = StandardScaler()
        
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
            df['Season'] = (df['Month'] % 12 + 3) // 3
            df['Temp_Range'] = df['Tx'] - df['Tn']
            
            # Add rolling features with careful handling of NaN values
            windows = [3, 7]
            for window in windows:
                # Rolling mean with min_periods=1 to avoid NaN
                df[f'RR_Rolling_Mean_{window}d'] = df[self.target_column].rolling(window=window, min_periods=1).mean()
                df[f'Tavg_Rolling_Mean_{window}d'] = df['Tavg'].rolling(window=window, min_periods=1).mean()
            
            # Add lag features (only lag 1 and 3 to reduce NaN issues)
            for lag in [1, 3]:
                df[f'RR_Lag_{lag}'] = df[self.target_column].shift(lag)
                df[f'Tavg_Lag_{lag}'] = df['Tavg'].shift(lag)
            
            # Drop rows with NaN values that couldn't be imputed
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Log any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                logging.info(f"\nRows dropped due to NaN values: {len(df_before_drop) - len(df)}")
            
            # Update feature columns with new features that were successfully created
            self.feature_columns = [
                'Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x',
                'Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range'
            ]
            
            # Add rolling and lag features if they exist and don't have NaN values
            for col in df.columns:
                if col.startswith(('RR_Rolling', 'Tavg_Rolling', 'RR_Lag', 'Tavg_Lag')) and not df[col].isna().any():
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
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'seasonal_patterns.png'))
        plt.close()

    def plot_predictions(self, dates, actual, predicted):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, actual, label='Actual', alpha=0.7)
        plt.plot(dates, predicted, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Rainfall')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'predictions.png'))
        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Plot top 20 features for readability
        top_n = min(20, len(self.feature_columns))
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(top_n), importance[indices][:top_n])
        plt.xticks(
            range(top_n), 
            [self.feature_columns[i] for i in indices][:top_n], 
            rotation=45, 
            ha='right'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_importance.png'))
        plt.close()

    def save_metrics(self, metrics):
        """Save metrics to a file"""
        metrics_file = os.path.join(self.run_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("="*50 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

    def train_and_evaluate(self, df):
        """Train and evaluate with simplified approach to avoid NaN errors"""
        try:
            # Prepare features and target - dropna as a safety measure
            df_clean = df.dropna()
            if len(df_clean) < len(df):
                logging.warning(f"Dropped {len(df) - len(df_clean)} rows with NaN values before training")
            
            X = df_clean[self.feature_columns].values
            y = df_clean[self.target_column].values
            dates = df_clean['Tanggal'].values
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Verify no NaN values exist in the data
            if np.isnan(X).any():
                logging.error("NaN values found in scaled features!")
                # Find which features have NaN values
                nan_cols = np.isnan(X).any(axis=0)
                for i, has_nan in enumerate(nan_cols):
                    if has_nan:
                        logging.error(f"Feature '{self.feature_columns[i]}' has NaN values")
                raise ValueError("Input contains NaN values. Fix preprocessing before training.")
            
            # Calculate split index for chronological split
            split_idx = int(len(X) * 0.8)
            
            # Split the data chronologically
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            dates_test = dates[split_idx:]
            
            # Log split information
            logging.info(f"\nTraining data shape: {X_train.shape}")
            logging.info(f"Testing data shape: {X_test.shape}")
            logging.info(f"Training date range: {dates[0]} to {dates[split_idx-1]}")
            logging.info(f"Testing date range: {dates[split_idx]} to {dates[-1]}")
            
            # Use a simpler model with good defaults instead of grid search
            logging.info("Training GradientBoostingRegressor...")
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }
            
            # Create visualizations
            self.plot_feature_correlations(df_clean)
            self.plot_feature_distributions(df_clean)
            self.plot_seasonal_patterns(df_clean)
            self.plot_predictions(dates_test, y_test, y_pred)
            self.plot_feature_importance()
            
            # Save metrics
            self.save_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise

    def save_model(self):
        """Save model"""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'timestamp': self.timestamp,
            'scaler': self.scaler
        }
        
        model_file = os.path.join(self.models_dir, f'gbm_model_{self.timestamp}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {model_file}")

def main():
    try:
        # Initialize predictor
        predictor = GBMWeatherPredictor()
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv('makassar.csv')
        
        # Log initial data info
        logging.info(f"\nInitial data shape: {df.shape}")
        
        # Preprocess data with outlier removal
        df_processed = predictor.preprocess_data(df)
        
        # Log processed data info
        logging.info(f"\nProcessed data shape: {df_processed.shape}")
        
        # Train and evaluate model
        metrics = predictor.train_and_evaluate(df_processed)
        
        # Save the model
        predictor.save_model()
        
        # Log results
        logging.info("\n" + "="*50)
        logging.info("MODEL PERFORMANCE METRICS")
        logging.info("="*50)
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 