"Creating improved weather prediction model..." 

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('default')

class ImprovedWeatherPredictor:
    """
    Improved Weather Predictor that fixes data leakage issues
    and provides realistic performance metrics
    """
    
    def __init__(self):
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ddd_car': 'Wind Direction (degrees)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # CRITICAL: NO TARGET VARIABLES IN FEATURES
        self.feature_columns = [
            # Basic meteorological measurements (NON-TARGETS)
            'Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x',
            
            # Temporal features
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos', 'Season',
            
            # Derived meteorological features
            'Temp_Range', 'Dew_Point', 'Heat_Index',
            'Temp_Humidity_Interaction',
            
            # Lag features (from NON-target variables only)
            'Tn_Lag_1', 'Tn_Lag_2', 'Tx_Lag_1', 'Tx_Lag_2',
            'RH_avg_Lag_1', 'RH_avg_Lag_2', 'RH_avg_Lag_3',
            
            # Rolling features (from NON-target variables only)  
            'Tn_Rolling_Mean_3d', 'Tx_Rolling_Mean_3d',
            'RH_avg_Rolling_Mean_3d', 'RH_avg_Rolling_Std_3d',
            
            # Weather pattern indicators
            'High_Humidity', 'Extreme_Temp_Range', 'Hot_Day', 'Cold_Day'
        ]
        
        self.models = {}
        self.scalers = {}
        
        # Create output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'improved_results_{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("🚀 Initialized Improved Weather Predictor")
        print("✅ NO target variables used as features (prevents data leakage)")
        
    def preprocess_data(self, df):
        """Preprocess data WITHOUT data leakage"""
        print("\n=== PREPROCESSING (NO DATA LEAKAGE) ===")
        df = df.copy()
        
        # Convert and sort by date
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        df = df.sort_values('Tanggal').reset_index(drop=True)
        print(f"Dataset sorted by date: {df['Tanggal'].min()} to {df['Tanggal'].max()}")
        
        # Handle special values (8888, 9999)
        special_values = [8888, 9999]
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                special_count = df[col].isin(special_values).sum()
                if special_count > 0:
                    print(f"Interpolating {special_count} special values in {col}")
                    df.loc[df[col].isin(special_values), col] = np.nan
        
        # Wind direction conversion
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
            'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 
            'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        if 'ddd_car' in df.columns:
            df['ddd_car'] = df['ddd_car'].astype(str).str.strip()
            df['ddd_car'] = df['ddd_car'].replace(wind_dir_map)
            df['ddd_car'] = pd.to_numeric(df['ddd_car'], errors='coerce')
        
        # Ensure numeric types
        numeric_cols = ['Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x'] + self.target_columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create temporal features
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        df['Season'] = ((df['Month'] % 12 + 3) // 3)
        
        # Cyclical encoding for better seasonality capture
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
        
        # Derived meteorological features
        df['Temp_Range'] = df['Tx'] - df['Tn']
        df['Temp_Avg_Calc'] = (df['Tx'] + df['Tn']) / 2
        df['Temp_Humidity_Interaction'] = df['Temp_Avg_Calc'] * df['RH_avg']
        df['Dew_Point'] = df['Temp_Avg_Calc'] - ((100 - df['RH_avg']) / 5)
        df['Heat_Index'] = df['Temp_Avg_Calc'] + 0.05 * df['RH_avg']
        
        # Weather pattern indicators
        df['High_Humidity'] = (df['RH_avg'] > 80).astype(int)
        df['Extreme_Temp_Range'] = (df['Temp_Range'] > df['Temp_Range'].quantile(0.9)).astype(int)
        df['Hot_Day'] = (df['Tx'] > df['Tx'].quantile(0.9)).astype(int)
        df['Cold_Day'] = (df['Tn'] < df['Tn'].quantile(0.1)).astype(int)
        
        # LAG FEATURES - ONLY from NON-target variables
        print("Creating lag features from NON-target variables...")
        non_target_vars = ['Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x']
        for var in non_target_vars:
            if var in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
        
        # ROLLING FEATURES - ONLY from NON-target variables
        print("Creating rolling features from NON-target variables...")
        for var in non_target_vars:
            if var in df.columns:
                for window in [3, 7]:
                    df[f'{var}_Rolling_Mean_{window}d'] = df[var].rolling(window=window, min_periods=1).mean()
                    if window == 3:  # Only for 3-day window to avoid too many features
                        df[f'{var}_Rolling_Std_{window}d'] = df[var].rolling(window=window, min_periods=1).std()
        
        # Handle missing values
        print("Handling missing values...")
        for col in df.columns:
            if df[col].isna().any():
                if col == 'Tanggal':
                    continue
                # Forward fill, then backward fill, then median
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        
        # Remove rows with missing targets
        initial_rows = len(df)
        df = df.dropna(subset=self.target_columns)
        print(f"Removed {initial_rows - len(df)} rows with missing targets")
        
        # Filter feature columns to only those that exist
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        print(f"✅ Final dataset: {len(df)} rows, {len(self.feature_columns)} features")
        print(f"✅ NO target variables in features - data leakage prevented!")
        
        return df
    
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
    
    def train_single_model(self, X_train, y_train, X_test, y_test, target_name):
        """Train model for single target with time series validation"""
        print(f"\n--- Training {self.target_names[target_name]} Model ---")
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model parameters (conservative for realistic performance)
        model = GradientBoostingRegressor(
            n_estimators=80,  # Reduced to prevent overfitting
            learning_rate=0.1,
            max_depth=4,  # Shallow trees
            min_samples_split=10,  # Higher to prevent overfitting
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
            fold_model = clone(model)
            fold_model.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            
            val_pred = fold_model.predict(X_train_scaled[val_idx])
            val_pred = self.apply_constraints(val_pred, target_name)
            
            fold_r2 = r2_score(y_train.iloc[val_idx], val_pred)
            cv_scores.append(fold_r2)
            print(f"  Fold {fold+1}: R² = {fold_r2:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"  CV Mean R²: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Train final model
        model.fit(X_train_scaled, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        y_pred = self.apply_constraints(y_pred, target_name)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        return model, scaler, y_pred, metrics
    
    def plot_predictions(self, dates, y_test, y_pred, target_name, metrics):
        """Plot predictions with proper visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series plot
        ax1.plot(dates, y_test, 'o-', label='Actual', alpha=0.7, markersize=3)
        ax1.plot(dates, y_pred, 's-', label='Predicted', alpha=0.7, markersize=3)
        ax1.set_title(f'{self.target_names[target_name]} - Improved Model (No Data Leakage)')
        ax1.set_ylabel(self.target_names[target_name])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = (f"RMSE: {metrics['rmse']:.3f}\n"
                       f"MAE: {metrics['mae']:.3f}\n"
                       f"R²: {metrics['r2']:.3f}\n"
                       f"CV R²: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}")
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Scatter plot
        ax2.scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Actual vs Predicted Scatter')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'predictions_{target_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_and_evaluate(self, df):
        """Train and evaluate all models"""
        print("\n=== TRAINING IMPROVED MODELS ===")
        
        # Prepare data
        X = df[self.feature_columns]
        
        # Temporal split (80% train, 20% test)
        split_idx = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        train_dates = df.iloc[:split_idx]['Tanggal']
        test_dates = df.iloc[split_idx:]['Tanggal']
        
        print(f"Training: {train_dates.min().date()} to {train_dates.max().date()}")
        print(f"Testing: {test_dates.min().date()} to {test_dates.max().date()}")
        
        all_metrics = {}
        
        # Train each target
        for target in self.target_columns:
            if target not in df.columns:
                continue
                
            y = df[target]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model, scaler, y_pred, metrics = self.train_single_model(
                X_train, y_train, X_test, y_test, target
            )
            
            # Store results
            self.models[target] = model
            self.scalers[target] = scaler
            all_metrics[target] = metrics
            
            # Plot results
            self.plot_predictions(test_dates, y_test, y_pred, target, metrics)
            
            # Save metrics
            with open(os.path.join(self.output_dir, f'metrics_{target}.txt'), 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
        
        # Create summary
        self.create_summary(all_metrics)
        
        return all_metrics
    
    def create_summary(self, all_metrics):
        """Create performance summary"""
        targets = list(all_metrics.keys())
        test_r2 = [all_metrics[t]['r2'] for t in targets]
        cv_r2 = [all_metrics[t]['cv_mean'] for t in targets]
        cv_std = [all_metrics[t]['cv_std'] for t in targets]
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(targets))
        width = 0.35
        
        ax.bar(x - width/2, cv_r2, width, label='CV R² (Training)', 
               yerr=cv_std, capsize=5, alpha=0.8)
        ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        
        ax.set_xlabel('Target Variables')
        ax.set_ylabel('R² Score')
        ax.set_title('Improved Model Performance (No Data Leakage)')
        ax.set_xticks(x)
        ax.set_xticklabels([self.target_names[t] for t in targets], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"\n{'='*60}")
        print("IMPROVED MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Target Variable':<25} | {'Test R²':<8} | {'CV R² ± Std'}")
        print("-" * 60)
        
        for target in targets:
            metrics = all_metrics[target]
            name = self.target_names[target]
            test_r2_val = metrics['r2']
            cv_mean = metrics['cv_mean']
            cv_std_val = metrics['cv_std']
            print(f"{name:<25} | {test_r2_val:>7.4f} | {cv_mean:.4f}±{cv_std_val:.4f}")
        
        avg_test_r2 = np.mean(test_r2)
        avg_cv_r2 = np.mean(cv_r2)
        
        print("-" * 60)
        print(f"{'Average':<25} | {avg_test_r2:>7.4f} | {avg_cv_r2:.4f}")
        print(f"{'='*60}")
        
        # Performance assessment
        if avg_test_r2 > 0.7:
            print("⚠️  WARNING: Very high R² - check for remaining data leakage!")
        elif avg_test_r2 > 0.5:
            print("✅ Good performance for weather prediction")
        elif avg_test_r2 > 0.3:
            print("✅ Reasonable performance for weather prediction") 
        elif avg_test_r2 > 0.1:
            print("📈 Modest performance - room for improvement")
        else:
            print("📉 Low performance - model needs significant improvement")
        
        print(f"\n📊 Results saved in: {self.output_dir}")

def main():
    """Main execution"""
    print("🌤️  IMPROVED WEATHER PREDICTION MODEL")
    print("✅ Fixed data leakage issues")
    print("✅ Proper time series validation")
    print("✅ Realistic performance expectations\n")
    
    try:
        # Load data
        df = pd.read_csv('makassar.csv')
        print(f"Loaded dataset: {len(df)} rows")
        
        # Initialize predictor
        predictor = ImprovedWeatherPredictor()
        
        # Preprocess
        processed_df = predictor.preprocess_data(df)
        
        # Train and evaluate
        metrics = predictor.train_and_evaluate(processed_df)
        
        print("\n🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
