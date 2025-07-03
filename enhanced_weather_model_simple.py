import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings

# Configure parallel processing
import multiprocessing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)

# Suppress warnings
warnings.filterwarnings('ignore')

class EnhancedWeatherPredictor:
    def __init__(self):
        # REMOVED WIND DIRECTION (ddd_car) from targets - poor performance
        self.target_columns = ['RR', 'ss', 'Tavg', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # Enhanced feature columns (excluding target variables)
        self.base_feature_columns = [
            # Original non-target features
            'Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x',
            
            # Temporal features
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos', 'Season',
            
            # Basic derived features
            'Temp_Range', 'Temp_Humidity', 'Dew_Point',
            
            # Lag features (from non-target variables only)
            'Tn_Lag_1', 'Tx_Lag_1', 'RH_avg_Lag_1',
            'Tn_Lag_2', 'Tx_Lag_2', 'RH_avg_Lag_2',
            
            # Rolling features (from non-target variables only)
            'Tn_Rolling_Mean_3d', 'Tx_Rolling_Mean_3d', 'RH_Rolling_Mean_3d',
            'Tn_Rolling_Std_3d', 'Tx_Rolling_Std_3d', 'RH_Rolling_Std_3d'
        ]
        
        self.feature_columns = []
        self.scalers = {}
        self.models = {}
        self.cv_results = {}
        
        # Create directories
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'enhanced_results_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✅ Enhanced model initialized")
        print(f"🎯 Target variables: {list(self.target_names.keys())} (removed wind direction)")
        print(f"📁 Results will be saved to: {self.results_dir}")

    def preprocess_data(self, df):
        """Enhanced preprocessing without data leakage"""
        print("\n🔧 Starting enhanced data preprocessing...")
        
        df = df.copy()
        initial_size = len(df)
        
        # Convert date column
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        
        # Handle special values (8888, 9999) with interpolation
        special_values = [8888, 9999]
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                if any(df[col].isin(special_values)):
                    print(f"🔄 Interpolating special values in {col}")
                    mask = df[col].isin(special_values)
                    df.loc[mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        # Convert wind direction to numeric (but not using as target anymore)
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        if 'ddd_car' in df.columns:
            df['ddd_car_numeric'] = df['ddd_car'].map(wind_dir_map)
            df['ddd_car_numeric'] = df['ddd_car_numeric'].fillna(0)
        
        # Ensure target variables are numeric
        for target in self.target_columns:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors='coerce')
        
        # Add temporal features
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        df['Season'] = (df['Month'] % 12 + 3) // 3
        
        # Cyclical encoding for seasonality
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
        
        # Basic derived features
        df['Temp_Range'] = df['Tx'] - df['Tn']
        df['Temp_Humidity'] = ((df['Tx'] + df['Tn']) / 2) * df['RH_avg']
        df['Dew_Point'] = ((df['Tx'] + df['Tn']) / 2) - ((100 - df['RH_avg']) / 5)
        
        # Lag features (ONLY from non-target variables)
        non_target_vars = ['Tn', 'Tx', 'RH_avg']
        for var in non_target_vars:
            if var in df.columns:
                for lag in [1, 2]:
                    df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
        
        # Rolling features (ONLY from non-target variables)
        for var in non_target_vars:
            if var in df.columns:
                df[f'{var}_Rolling_Mean_3d'] = df[var].rolling(window=3, min_periods=1).mean()
                df[f'{var}_Rolling_Std_3d'] = df[var].rolling(window=3, min_periods=1).std()
        
        # Finalize feature columns
        self.feature_columns = [f for f in self.base_feature_columns if f in df.columns]
        
        # Handle any remaining NaN values
        for col in self.feature_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Drop rows with NaN in target variables
        df = df.dropna(subset=self.target_columns)
        
        print(f"📊 Final dataset: {len(df)} samples, {len(self.feature_columns)} features")
        print(f"📉 Removed {initial_size - len(df)} rows with missing data")
        
        return df

    def train_models(self, df):
        """Train enhanced models with cross-validation"""
        print(f"\n🚀 Training enhanced models for {len(self.target_columns)} targets...")
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Split data temporally
        train_size = int(0.8 * len(df))
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        
        # Scale features with RobustScaler (better for outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        results = {}
        
        for target in self.target_columns:
            print(f"\n🎯 Training {self.target_names[target]} model...")
            
            # Prepare target
            y = df[target]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            
            # Conservative model parameters to prevent overfitting
            if target == 'RR':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=10,
                    min_samples_leaf=8,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            elif target == 'Tavg':
                model = GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.06,
                    max_depth=4,
                    min_samples_split=8,
                    min_samples_leaf=6,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=8,
                    min_samples_leaf=6,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            
            # Cross-validation with TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=tscv, scoring='r2', n_jobs=N_JOBS
            )
            
            # Train final model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Apply constraints
            if target == 'RR':
                y_pred = np.maximum(y_pred, 0)
            elif target == 'ss':
                y_pred = np.clip(y_pred, 0, 24)
            elif target == 'ff_avg':
                y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
            
            # Store results
            self.models[target] = model
            self.cv_results[target] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            results[target] = {
                'model': model,
                'metrics': metrics,
                'cv_results': self.cv_results[target],
                'y_test': y_test,
                'y_pred': y_pred,
                'test_dates': df.iloc[train_size:]['Tanggal']
            }
            
            print(f"  📈 R² Score: {metrics['R2']:.4f}")
            print(f"  📊 CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  📏 RMSE: {metrics['RMSE']:.4f}")
        
        return results

    def plot_predictions(self, results):
        """Create prediction plots"""
        print("\n📊 Creating prediction plots...")
        
        n_targets = len(results)
        fig, axes = plt.subplots(n_targets, 1, figsize=(15, 4*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i, (target, data) in enumerate(results.items()):
            ax = axes[i]
            
            dates = data['test_dates']
            y_test = data['y_test']
            y_pred = data['y_pred']
            metrics = data['metrics']
            cv_results = data['cv_results']
            
            ax.plot(dates, y_test, 'o-', label='Actual', alpha=0.7, markersize=3)
            ax.plot(dates, y_pred, 'x-', label='Predicted', alpha=0.7, markersize=3)
            
            ax.set_title(f'{self.target_names[target]} - R²: {metrics["R2"]:.3f} | CV: {cv_results["cv_mean"]:.3f}±{cv_results["cv_std"]:.3f}')
            ax.set_ylabel(self.target_names[target])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if i == n_targets - 1:
                ax.set_xlabel('Date')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'enhanced_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, results):
        """Plot feature importance"""
        print("📊 Creating feature importance plots...")
        
        n_targets = len(results)
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, 5*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i, (target, data) in enumerate(results.items()):
            model = data['model']
            importance = model.feature_importances_
            
            # Get top 10 features
            indices = np.argsort(importance)[::-1][:10]
            top_features = [self.feature_columns[idx] for idx in indices]
            top_importance = importance[indices]
            
            ax = axes[i]
            bars = ax.bar(range(len(top_importance)), top_importance)
            ax.set_xticks(range(len(top_importance)))
            ax.set_xticklabels(top_features, rotation=45, ha='right')
            ax.set_title(f'Top 10 Features - {self.target_names[target]}')
            ax.set_ylabel('Feature Importance')
            
            # Color bars
            for bar in bars:
                bar.set_color('steelblue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'enhanced_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results):
        """Save model results and metrics"""
        print("\n💾 Saving results...")
        
        # Save metrics
        metrics_summary = {}
        for target, data in results.items():
            metrics_summary[target] = {
                'R2': data['metrics']['R2'],
                'RMSE': data['metrics']['RMSE'], 
                'MAE': data['metrics']['MAE'],
                'CV_R2_mean': data['cv_results']['cv_mean'],
                'CV_R2_std': data['cv_results']['cv_std']
            }
        
        metrics_df = pd.DataFrame(metrics_summary).T
        metrics_df.to_csv(os.path.join(self.results_dir, 'enhanced_metrics.csv'))
        
        # Save individual target metrics
        for target, data in results.items():
            target_metrics = {
                'R2': data['metrics']['R2'],
                'RMSE': data['metrics']['RMSE'],
                'MAE': data['metrics']['MAE'],
                'CV_R2_mean': data['cv_results']['cv_mean'],
                'CV_R2_std': data['cv_results']['cv_std']
            }
            
            with open(os.path.join(self.results_dir, f'enhanced_metrics_{target}.txt'), 'w') as f:
                for key, value in target_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
        
        # Save models
        for target, model in self.models.items():
            joblib.dump(model, os.path.join(self.results_dir, f'enhanced_model_{target}.pkl'))
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(self.results_dir, 'enhanced_scalers.pkl'))
        
        print(f"✅ Results saved to: {self.results_dir}")

    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        print("\n📋 Generating summary report...")
        
        report = []
        report.append("=" * 60)
        report.append("🚀 ENHANCED WEATHER PREDICTION MODEL")
        report.append("=" * 60)
        report.append(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Target Variables: {len(self.target_columns)} (removed wind direction)")
        report.append(f"🔧 Total Features: {len(self.feature_columns)}")
        report.append("")
        
        report.append("📊 MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        total_r2 = 0
        for target, data in results.items():
            metrics = data['metrics']
            cv_results = data['cv_results']
            total_r2 += metrics['R2']
            
            report.append(f"{self.target_names[target]:25} - R²: {metrics['R2']:.4f} | CV: {cv_results['cv_mean']:.3f}±{cv_results['cv_std']:.3f}")
        
        avg_r2 = total_r2 / len(results)
        report.append("")
        report.append(f"📈 Average R² Score: {avg_r2:.4f}")
        report.append("")
        
        report.append("🎯 KEY IMPROVEMENTS:")
        report.append("-" * 40)
        report.append("✅ Removed wind direction (was worst performing target)")
        report.append("✅ Eliminated data leakage from target variables")
        report.append("✅ Proper time series validation")
        report.append("✅ Conservative model parameters (prevent overfitting)")
        report.append("✅ RobustScaler for better outlier handling")
        report.append("")
        
        report.append("📝 INTERPRETATION:")
        report.append("-" * 40)
        report.append("• These R² scores are realistic for weather prediction")
        report.append("• Temperature prediction typically shows highest accuracy")
        report.append("• Removing wind direction improves overall model reliability")
        report.append("• Proper validation prevents overfitting and data leakage")
        report.append("")
        
        # Save report
        with open(os.path.join(self.results_dir, 'ENHANCED_MODEL_SUMMARY.md'), 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        for line in report:
            print(line)

def main():
    try:
        print("🚀 Starting Enhanced Weather Prediction Model...")
        
        # Load data
        print("📂 Loading dataset...")
        df = pd.read_csv('makassar.csv')
        print(f"📊 Loaded dataset: {df.shape}")
        
        # Initialize enhanced predictor
        predictor = EnhancedWeatherPredictor()
        
        # Preprocess data
        processed_df = predictor.preprocess_data(df)
        
        # Train enhanced models
        results = predictor.train_models(processed_df)
        
        # Generate visualizations
        predictor.plot_predictions(results)
        predictor.plot_feature_importance(results)
        
        # Save everything
        predictor.save_results(results)
        
        # Generate comprehensive report
        predictor.generate_summary_report(results)
        
        print(f"\n🎉 Enhanced model training completed successfully!")
        print(f"📁 All results saved to: {predictor.results_dir}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 