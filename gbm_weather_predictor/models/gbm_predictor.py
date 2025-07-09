"""
Main GBM Weather Predictor module
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
import time
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm

from ..config.settings import TARGET_COLUMNS, TARGET_NAMES, MODEL_CONFIGS, MODELS_DIR, PLOTS_DIR
from ..data.preprocessing import DataPreprocessor
from ..data.feature_engineering import FeatureEngineer
from ..visualization.plotting import WeatherPlotter
from ..visualization.journal_plotter import JournalPlotter

class GBMWeatherPredictor:
    def __init__(self):
        self.model = None
        self.multi_day_models = {}  # For storing models for different forecast horizons
        self.multi_target_models = {}  # For storing models for different target variables
        
        self.scaler = StandardScaler()
        self.target_scalers = {}  # Separate scalers for each target
        self.forecast_days = 5  # Number of days to forecast ahead
        self.cv_folds = 5  # Number of cross-validation folds
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.plotter = WeatherPlotter(self.run_dir)
        self.journal_plotter = JournalPlotter("journal_plots")

    def train_and_evaluate(self, df):
        """Train and evaluate GBM models for all target variables"""
        try:
            print(f"\n=== Training Multi-Target Weather Prediction Models ===")
            print(f"Target variables: {', '.join([TARGET_NAMES[t] for t in TARGET_COLUMNS])}")
            
            # Prepare feature matrix
            X = df[self.feature_engineer.feature_columns]
            print(f"\nFeature matrix shape: {X.shape}")
            print(f"Using {len(self.feature_engineer.feature_columns)} features")
            
            # Split into training and testing sets (temporal split)
            train_size = int(0.8 * len(df))
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:]
            
            # Scale features once
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store for later use
            self.X_train = X_train
            self.X_train_scaled = X_train_scaled
            
            # Train separate models for each target variable
            all_metrics = {}
            
            for target in TARGET_COLUMNS:
                if target not in df.columns:
                    print(f"\nSkipping {target} - not found in dataset")
                    continue
                    
                print(f"\n--- Training model for {TARGET_NAMES[target]} ({target}) ---")
                
                # Prepare target variable
                y = df[target]
                y_train = y.iloc[:train_size]
                y_test = y.iloc[train_size:]
                
                # Scale target if needed (for regression targets)
                target_scaler = StandardScaler()
                if target in ['RR', 'ss', 'ff_avg']:  # These benefit from scaling
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    self.target_scalers[target] = target_scaler
                else:
                    y_train_scaled = y_train.values
                    self.target_scalers[target] = None
                
                # Define model for this target
                model_config = MODEL_CONFIGS.get(target, MODEL_CONFIGS['RR'])
                model = GradientBoostingRegressor(**model_config)
                
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
                if target == 'RR':  # Rainfall cannot be negative
                    y_pred = np.maximum(y_pred, 0)
                elif target == 'ss':  # Sunshine hours: 0-24 hours
                    y_pred = np.clip(y_pred, 0, 24)
                elif target == 'ff_avg':  # Wind speed cannot be negative
                    y_pred = np.maximum(y_pred, 0)
                elif target == 'ddd_car':  # Wind direction: 0-360 degrees
                    y_pred = np.clip(y_pred % 360, 0, 360)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate additional metrics specific to target
                if target == 'RR':
                    # Rain-specific metrics
                    rain_days_actual = (y_test > 0).sum()
                    rain_days_predicted = (y_pred > 0).sum()
                    rain_detection_accuracy = np.mean((y_test > 0) == (y_pred > 0))
                    
                    print(f"Rain Detection Accuracy: {rain_detection_accuracy:.4f}")
                    print(f"Actual rain days: {rain_days_actual}, Predicted: {rain_days_predicted}")
                
                # Print metrics
                print(f"\n{TARGET_NAMES[target]} Model Evaluation:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
                
                # Store model and results
                self.multi_target_models[target] = {
                    'model': model,
                    'scaler': self.target_scalers[target],
                    'predictions': y_pred,
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
                test_dates = df.iloc[train_size:]['Tanggal'] if 'Tanggal' in df.columns else range(len(y_test))
                self.plotter.plot_predictions(test_dates, y_test, y_pred, TARGET_NAMES[target])
                
                # Store for summary
                all_metrics[target] = self.multi_target_models[target]['metrics']
            
            # Print summary of all models
            print(f"\n=== Multi-Target Model Summary ===")
            for target, metrics in all_metrics.items():
                print(f"{TARGET_NAMES[target]:25} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Plot comprehensive visualizations
            print(f"\nGenerating visualization plots...")
            self.plotter.plot_multi_target_feature_importance(self.multi_target_models, self.feature_engineer.feature_columns)
            self.plotter.plot_feature_correlations(df, self.feature_engineer.feature_columns)
            self.plotter.plot_feature_distributions(df, self.feature_engineer.feature_columns)
            self.plotter.plot_seasonal_patterns(df)
            
            # Generate journal-quality plots
            print(f"\n🎨 Generating Journal-Quality Plots...")
            test_data = {
                'dates': df.iloc[train_size:]['Tanggal'] if 'Tanggal' in df.columns else range(len(y_test)),
                'actual': pd.DataFrame({target: df.iloc[train_size:][target] for target in TARGET_COLUMNS if target in df.columns}),
                'predicted': pd.DataFrame({target: self.multi_target_models[target]['predictions'] for target in TARGET_COLUMNS if target in self.multi_target_models})
            }
            
            self.journal_plotter.create_comprehensive_journal_plots(
                df, 
                self.multi_target_models, 
                self.feature_engineer.feature_columns,
                test_data
            )
            
            return all_metrics
            
        except Exception as e:
            print(f"Error in model training and evaluation: {str(e)}")
            raise

    def save_metrics(self, metrics, target_name=''):
        """Save evaluation metrics to a file"""
        filename = f'metrics_{target_name}.txt' if target_name else 'metrics.txt'
        with open(os.path.join(self.run_dir, filename), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def save_multi_target_models(self):
        """Save all multi-target models to disk"""
        # Save main models
        for target, model_data in self.multi_target_models.items():
            target_dir = os.path.join(self.models_dir, f'target_{target}')
            os.makedirs(target_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(target_dir, 'gbm_model.pkl')
            joblib.dump(model_data['model'], model_path)
            
            # Save scaler if exists
            if model_data['scaler'] is not None:
                scaler_path = os.path.join(target_dir, 'target_scaler.pkl')
                joblib.dump(model_data['scaler'], scaler_path)
            
            print(f"Saved {TARGET_NAMES[target]} model in {target_dir}")
        
        # Save main feature scaler and feature list
        main_scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, main_scaler_path)
        
        features_path = os.path.join(self.models_dir, 'feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_engineer.feature_columns, f)
            
        print(f"\nSaved feature scaler and feature list in {self.models_dir}")

    def save_model(self):
        """Save the trained models - updated for multi-target"""
        if not self.multi_target_models:
            print("Models not trained yet, nothing to save.")
            return
            
        self.save_multi_target_models() 