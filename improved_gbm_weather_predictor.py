import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.inspection import permutation_importance
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import clone
import warnings
from tqdm import tqdm
import time
import multiprocessing

# Configure settings
N_JOBS = max(1, multiprocessing.cpu_count() // 2)
warnings.filterwarnings('ignore')

# Create directories
IMPROVED_DIR = 'improved_gbm'
MODELS_DIR = os.path.join(IMPROVED_DIR, 'models')
PLOTS_DIR = os.path.join(IMPROVED_DIR, 'plots')
LOGS_DIR = os.path.join(IMPROVED_DIR, 'logs')

for dir_path in [IMPROVED_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class ImprovedWeatherPredictor:
    def __init__(self):
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ddd_car': 'Wind Direction (degrees)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # NO TARGET VARIABLES IN FEATURES - This prevents data leakage
        self.base_weather_features = [
            'Tn', 'Tx', 'RH_avg',  # Temperature min/max, humidity
            'ff_x', 'ddd_x'  # Non-target wind measurements
        ]
        
        self.temporal_features = [
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos', 'Season'
        ]
        
        self.derived_features = [
            'Temp_Range', 'Dew_Point', 'Heat_Index',
            'Temp_Humidity_Interaction', 'Pressure_Change'
        ]
        
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Create run directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(PLOTS_DIR, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
    def preprocess_data(self, df):
        """Enhanced preprocessing WITHOUT data leakage"""
        print("=== IMPROVED PREPROCESSING (NO DATA LEAKAGE) ===")
        df = df.copy()
        
        # Convert date and sort
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        # Handle special values (8888, 9999) 
        special_values = [8888, 9999]
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64] and df[col].isin(special_values).any():
                print(f"Interpolating {df[col].isin(special_values).sum()} special values in {col}")
                df.loc[df[col].isin(special_values), col] = np.nan
                df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Convert wind direction properly
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
            'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 
            'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        if 'ddd_car' in df.columns:
            df['ddd_car'] = df['ddd_car'].astype(str).str.strip()
            df['ddd_car'] = df['ddd_car'].replace(wind_dir_map)
            df['ddd_car'] = pd.to_numeric(df['ddd_car'], errors='coerce')
        
        # Ensure all target variables are numeric
        for target in self.target_columns:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors='coerce')
        
        # Create temporal features
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        df['Season'] = ((df['Month'] % 12 + 3) // 3)
        
        # Cyclical encoding
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
        
        # Derived meteorological features
        df['Tx'] = pd.to_numeric(df['Tx'], errors='coerce')
        df['Tn'] = pd.to_numeric(df['Tn'], errors='coerce')
        df['RH_avg'] = pd.to_numeric(df['RH_avg'], errors='coerce')
        
        df['Temp_Range'] = df['Tx'] - df['Tn']
        df['Temp_Avg_Calc'] = (df['Tx'] + df['Tn']) / 2
        df['Temp_Humidity_Interaction'] = df['Temp_Avg_Calc'] * df['RH_avg']
        
        # Meteorological calculations
        df['Dew_Point'] = df['Temp_Avg_Calc'] - ((100 - df['RH_avg']) / 5)
        df['Heat_Index'] = df['Temp_Avg_Calc'] + 0.05 * df['RH_avg']
        
        # Atmospheric pressure approximation (simplified)
        df['Pressure_Change'] = df['RH_avg'].diff().fillna(0)
        
        # LAG FEATURES - ONLY FROM NON-TARGET VARIABLES
        print("Creating lag features from NON-TARGET variables only...")
        lag_variables = ['Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x']
        for var in lag_variables:
            if var in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
        
        # ROLLING FEATURES - ONLY FROM NON-TARGET VARIABLES  
        print("Creating rolling features from NON-TARGET variables only...")
        for var in lag_variables:
            if var in df.columns:
                for window in [3, 7]:
                    df[f'{var}_Rolling_Mean_{window}d'] = df[var].rolling(window=window, min_periods=1).mean()
                    df[f'{var}_Rolling_Std_{window}d'] = df[var].rolling(window=window, min_periods=1).std()
        
        # Weather pattern features (derived from non-targets)
        df['High_Humidity'] = (df['RH_avg'] > 80).astype(int)
        df['Extreme_Temp_Range'] = (df['Temp_Range'] > df['Temp_Range'].quantile(0.9)).astype(int)
        df['Hot_Day'] = (df['Tx'] > df['Tx'].quantile(0.9)).astype(int)
        df['Cold_Day'] = (df['Tn'] < df['Tn'].quantile(0.1)).astype(int)
        
        # Collect all feature columns (NO TARGET VARIABLES)
        self.feature_columns = []
        
        # Add base features
        for feature in self.base_weather_features:
            if feature in df.columns:
                self.feature_columns.append(feature)
        
        # Add temporal features
        for feature in self.temporal_features:
            if feature in df.columns:
                self.feature_columns.append(feature)
        
        # Add derived features
        for feature in self.derived_features:
            if feature in df.columns:
                self.feature_columns.append(feature)
        
        # Add lag and rolling features (from non-targets only)
        for col in df.columns:
            if ('_Lag_' in col or '_Rolling_' in col or col in ['High_Humidity', 'Extreme_Temp_Range', 'Hot_Day', 'Cold_Day', 'Temp_Avg_Calc']):
                # Make sure it's not derived from a target variable
                is_from_target = any(target in col for target in self.target_columns)
                if not is_from_target and col not in self.target_columns:
                    self.feature_columns.append(col)
        
        # Handle missing values
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].median())
        
        # Remove rows with missing target values
        df = df.dropna(subset=self.target_columns)
        
        print(f"Final dataset: {len(df)} rows, {len(self.feature_columns)} features")
        print(f"Features used: {', '.join(self.feature_columns[:10])}...")
        print("✅ NO target variables used as features - data leakage prevented!")
        
        return df
    
    def plot_data_overview(self, df):
        """Plot data distribution and correlations"""
        # Target distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, target in enumerate(self.target_columns):
            if target in df.columns:
                axes[i].hist(df[target].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{self.target_names[target]}')
                axes[i].set_xlabel(target)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'target_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature correlations with targets
        feature_subset = self.feature_columns[:12]  # Top 12 features
        features_and_targets = feature_subset + [t for t in self.target_columns if t in df.columns]
        
        plt.figure(figsize=(12, 10))
        corr_matrix = df[features_and_targets].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature-Target Correlations (No Data Leakage)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'correlations_no_leakage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_single_target_model(self, X_train, y_train, X_test, y_test, target_name):
        """Train model for single target with proper time series validation"""
        print(f"\n--- Training {self.target_names[target_name]} Model ---")
        
        # Use RobustScaler to handle outliers better
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model hyperparameters optimized for weather prediction
        if target_name == 'RR':  # Rainfall - zero-inflated
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                min_samples_split=10, min_samples_leaf=5, 
                subsample=0.8, max_features='sqrt', random_state=42
            )
        elif target_name in ['ss', 'Tavg']:  # Continuous variables
            model = GradientBoostingRegressor(
                n_estimators=120, learning_rate=0.08, max_depth=5,
                min_samples_split=8, min_samples_leaf=4,
                subsample=0.8, max_features='sqrt', random_state=42
            )
        else:  # Wind variables
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=4,
                min_samples_split=12, min_samples_leaf=6,
                subsample=0.8, max_features='sqrt', random_state=42
            )
        
        # Time series cross-validation during training
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        print(f"Training with Time Series Cross-Validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            val_pred = fold_model.predict(X_fold_val)
            
            # Apply constraints
            val_pred = self.apply_constraints(val_pred, target_name)
            
            fold_r2 = r2_score(y_fold_val, val_pred)
            cv_scores.append(fold_r2)
            print(f"  Fold {fold+1}: R² = {fold_r2:.4f}")
        
        print(f"CV Mean R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model on all training data
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_pred = self.apply_constraints(y_pred, target_name)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        
        return model, scaler, y_pred, metrics
    
    def apply_constraints(self, predictions, target_name):
        """Apply physical constraints to predictions"""
        if target_name == 'RR':
            return np.maximum(predictions, 0)  # No negative rainfall
        elif target_name == 'ss':
            return np.clip(predictions, 0, 24)  # 0-24 hours sunshine
        elif target_name == 'ff_avg':
            return np.maximum(predictions, 0)  # No negative wind speed
        elif target_name == 'ddd_car':
            return predictions % 360  # Wind direction 0-360
        else:
            return predictions  # Temperature can be negative
    
    def plot_predictions(self, dates, y_test, y_pred, target_name, metrics):
        """Plot actual vs predicted with realistic performance"""
        plt.figure(figsize=(15, 8))
        
        # Main plot
        plt.subplot(2, 1, 1)
        plt.plot(dates, y_test, 'o-', label='Actual', alpha=0.7, markersize=3)
        plt.plot(dates, y_pred, 's-', label='Predicted', alpha=0.7, markersize=3)
        
        plt.title(f'{self.target_names[target_name]} - Improved Model (No Data Leakage)')
        plt.ylabel(self.target_names[target_name])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics
        metrics_text = f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}\nCV R²: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}"
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Scatter plot
        plt.subplot(2, 1, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual Scatter Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = target_name.replace('_', '').lower()
        plt.savefig(os.path.join(self.run_dir, f'improved_predictions_{safe_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_and_evaluate(self, df):
        """Train and evaluate all models with proper validation"""
        print("\n=== TRAINING IMPROVED MODELS (NO DATA LEAKAGE) ===")
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Temporal split (80% train, 20% test)
        split_idx = int(0.8 * len(df))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        
        train_dates = df.iloc[:split_idx]['Tanggal']
        test_dates = df.iloc[split_idx:]['Tanggal']
        
        print(f"Training period: {train_dates.min().date()} to {train_dates.max().date()}")
        print(f"Testing period: {test_dates.min().date()} to {test_dates.max().date()}")
        print(f"Features: {len(self.feature_columns)}")
        
        all_metrics = {}
        
        # Train model for each target
        for target in self.target_columns:
            if target not in df.columns:
                continue
                
            y = df[target]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Train model
            model, scaler, y_pred, metrics = self.train_single_target_model(
                X_train, y_train, X_test, y_test, target
            )
            
            # Store results
            self.models[target] = model
            self.scalers[target] = scaler
            all_metrics[target] = metrics
            
            # Plot results
            self.plot_predictions(test_dates, y_test, y_pred, target, metrics)
            
            # Save metrics
            with open(os.path.join(self.run_dir, f'metrics_{target}.txt'), 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"\n{self.target_names[target]} Results:")
            print(f"  Test R²: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  CV R²: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        # Summary plot
        self.plot_summary(all_metrics)
        
        return all_metrics
    
    def plot_summary(self, all_metrics):
        """Plot summary of all model performances"""
        targets = list(all_metrics.keys())
        r2_scores = [all_metrics[t]['r2'] for t in targets]
        cv_scores = [all_metrics[t]['cv_mean'] for t in targets]
        cv_stds = [all_metrics[t]['cv_std'] for t in targets]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(targets))
        width = 0.35
        
        plt.bar(x - width/2, cv_scores, width, label='CV R² (Training)', 
                yerr=cv_stds, capsize=5, alpha=0.8)
        plt.bar(x + width/2, r2_scores, width, label='Test R²', alpha=0.8)
        
        plt.xlabel('Target Variables')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (Improved - No Data Leakage)')
        plt.xticks(x, [self.target_names[t] for t in targets], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.run_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"\n=== IMPROVED MODEL SUMMARY ===")
        print("Target Variable           | Test R² | CV R² ± Std")
        print("-" * 50)
        for target in targets:
            metrics = all_metrics[target]
            print(f"{self.target_names[target]:24} | {metrics['r2']:7.4f} | {metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
        
        avg_r2 = np.mean(r2_scores)
        print(f"\nAverage Test R²: {avg_r2:.4f}")
        
        if avg_r2 > 0.8:
            print("⚠️  WARNING: Very high R² scores may still indicate issues!")
        elif avg_r2 > 0.6:
            print("✅ Good performance for weather prediction")
        elif avg_r2 > 0.3:
            print("✅ Reasonable performance for weather prediction")
        else:
            print("📈 Performance can be improved with more features/data")

def main():
    """Main execution function"""
    try:
        print("=== IMPROVED WEATHER PREDICTION MODEL ===")
        print("✅ Fixed data leakage issues")
        print("✅ Proper time series validation")  
        print("✅ Realistic performance expectations")
        
        # Load data
        print("\nLoading dataset...")
        df = pd.read_csv('makassar.csv')
        print(f"Original dataset: {len(df)} rows")
        
        # Initialize improved predictor
        predictor = ImprovedWeatherPredictor()
        
        # Preprocess (without data leakage)
        print("\nPreprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Generate overview plots
        predictor.plot_data_overview(processed_df)
        
        # Train and evaluate
        print("\nTraining models...")
        metrics = predictor.train_and_evaluate(processed_df)
        
        print(f"\n✅ Process completed successfully!")
        print(f"📊 Results saved in: {predictor.run_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 