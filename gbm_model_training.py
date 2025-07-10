import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from datetime import datetime
from gbm_config import *
from gbm_visualization import GBMVisualizer

class GBMModelTrainer:
    """Handles model training, evaluation, and prediction for the GBM weather prediction system"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scalers = {}  # Separate scalers for each target
        self.multi_target_models = {}  # For storing models for different target variables
        self.multi_day_models = {}  # For storing models for different forecast horizons
        self.visualizer = GBMVisualizer()
        
        # Save paths
        self.models_dir = MODELS_DIR
        self.plots_dir = PLOTS_DIR
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
    
    def save_metrics(self, metrics, target_name=''):
        """Save metrics to file"""
        metrics_file = os.path.join(self.run_dir, f'metrics_{target_name}.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Metrics for {target_name}\n")
            f.write("=" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
    
    def train_and_evaluate(self, df):
        """Main training and evaluation function"""
        try:
            print("Starting model training and evaluation...")
            
            # Prepare features and target
            available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            X = df[available_features]
            
            # Split data (80% train, 20% test)
            train_size = int(len(df) * 0.8)
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:]
            
            # Scale features
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
                if target in TARGETS_REQUIRING_SCALING:  # These benefit from scaling
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    self.target_scalers[target] = target_scaler
                else:
                    y_train_scaled = y_train.values
                    self.target_scalers[target] = None
                
                # Define model for this target
                model_config = MODEL_CONFIGS.get(target, MODEL_CONFIGS['default'])
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
                y_pred = self._apply_target_constraints(y_pred, target)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, target)
                
                # Print metrics
                print(f"\n{TARGET_NAMES[target]} Model Evaluation:")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAE: {metrics['MAE']:.4f}")
                print(f"  R²: {metrics['R2']:.4f}")
                
                # Store model and results
                self.multi_target_models[target] = {
                    'model': model,
                    'scaler': self.target_scalers[target],
                    'metrics': metrics
                }
                
                # Save metrics for this target
                self.save_metrics(metrics, target)
                
                # Plot predictions for this target
                test_dates = df.iloc[train_size:]['Tanggal'] if 'Tanggal' in df.columns else range(len(y_test))
                self.visualizer.plot_predictions(test_dates, y_test, y_pred, TARGET_NAMES[target])
                
                # Store for summary
                all_metrics[target] = metrics
            
            # Print summary of all models
            print(f"\n=== Multi-Target Model Summary ===")
            for target, metrics in all_metrics.items():
                print(f"{TARGET_NAMES[target]:25} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Plot comprehensive visualizations
            print(f"\nGenerating visualization plots...")
            self.visualizer.plot_feature_importance(self.multi_target_models)
            self.visualizer.plot_feature_correlations(df)
            self.visualizer.plot_feature_distributions(df)
            self.visualizer.plot_seasonal_patterns(df)
            self.visualizer.plot_data_splits(df)
            
            return all_metrics
            
        except Exception as e:
            print(f"Error in model training and evaluation: {str(e)}")
            raise
    
    def _apply_target_constraints(self, y_pred, target):
        """Apply constraints based on target type"""
        if target == 'RR':  # Rainfall cannot be negative
            y_pred = np.maximum(y_pred, 0)
        elif target == 'ss':  # Sunshine hours: 0-24 hours
            y_pred = np.clip(y_pred, 0, 24)
        elif target == 'ff_avg':  # Wind speed cannot be negative
            y_pred = np.maximum(y_pred, 0)
        elif target == 'ddd_car':  # Wind direction: 0-360 degrees
            y_pred = np.clip(y_pred % 360, 0, 360)
        
        return y_pred
    
    def _calculate_metrics(self, y_test, y_pred, target):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        # Calculate additional metrics specific to target
        if target == 'RR':
            # Rain-specific metrics
            rain_days_actual = (y_test > 0).sum()
            rain_days_predicted = (y_pred > 0).sum()
            rain_detection_accuracy = np.mean((y_test > 0) == (y_pred > 0))
            
            metrics['Rain_Detection_Accuracy'] = rain_detection_accuracy
            metrics['Actual_Rain_Days'] = rain_days_actual
            metrics['Predicted_Rain_Days'] = rain_days_predicted
            
            print(f"Rain Detection Accuracy: {rain_detection_accuracy:.4f}")
            print(f"Actual rain days: {rain_days_actual}, Predicted: {rain_days_predicted}")
        
        return metrics
    
    def train_multi_day_models(self, multi_day_data):
        """Train models for multi-day forecasting"""
        try:
            print("Training multi-day forecasting models...")
            
            results = {}
            
            for target, data in multi_day_data.items():
                if target not in TARGET_COLUMNS:
                    continue
                    
                print(f"\n--- Training multi-day models for {TARGET_NAMES[target]} ---")
                
                results[target] = {}
                
                # Get feature columns (exclude target columns)
                feature_cols = [col for col in data.columns if not col.startswith(f'{target}_Day_')]
                X = data[feature_cols]
                
                # Train models for each forecast day
                for day in range(1, FORECAST_DAYS + 1):
                    target_col = f'{target}_Day_{day}'
                    
                    if target_col not in data.columns:
                        print(f"Skipping {target_col} - not found in dataset")
                        continue
                    
                    print(f"Training model for day {day}...")
                    
                    y = data[target_col]
                    
                    # Remove rows with NaN values
                    valid_mask = ~(X.isna().any(axis=1) | y.isna())
                    X_valid = X[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(X_valid) == 0:
                        print(f"No valid data for {target_col}")
                        continue
                    
                    # Split data
                    train_size = int(len(X_valid) * 0.8)
                    X_train = X_valid.iloc[:train_size]
                    X_test = X_valid.iloc[train_size:]
                    y_train = y_valid.iloc[:train_size]
                    y_test = y_valid.iloc[train_size:]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Scale target if needed
                    target_scaler = StandardScaler()
                    if target in TARGETS_REQUIRING_SCALING:
                        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    else:
                        y_train_scaled = y_train.values
                        target_scaler = None
                    
                    # Train model
                    model_config = MODEL_CONFIGS.get(target, MODEL_CONFIGS['default'])
                    model = GradientBoostingRegressor(**model_config)
                    model.fit(X_train_scaled, y_train_scaled)
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_test_scaled)
                    
                    # Inverse transform if scaling was applied
                    if target_scaler is not None:
                        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    else:
                        y_pred = y_pred_scaled
                    
                    # Apply constraints
                    y_pred = self._apply_target_constraints(y_pred, target)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, target)
                    
                    # Store results
                    results[target][day] = {
                        'model': model,
                        'scaler': target_scaler,
                        'feature_scaler': scaler,
                        'metrics': metrics,
                        'dates': X_test.index,
                        'actual': y_test.values,
                        'predicted': y_pred
                    }
                    
                    print(f"  Day {day} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Plot multi-day prediction results
            self.visualizer.plot_multi_day_predictions(results)
            self.visualizer.plot_multi_day_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Error in multi-day model training: {str(e)}")
            raise
    
    def save_multi_target_models(self):
        """Save all trained models"""
        try:
            if not self.multi_target_models:
                print("No models to save")
                return
            
            # Create target-specific directories
            for target in self.multi_target_models.keys():
                target_dir = os.path.join(self.models_dir, f'target_{target}')
                os.makedirs(target_dir, exist_ok=True)
                
                # Save model
                model_path = os.path.join(target_dir, f'{target}_model.pkl')
                joblib.dump(self.multi_target_models[target]['model'], model_path)
                
                # Save scaler
                if self.multi_target_models[target]['scaler'] is not None:
                    scaler_path = os.path.join(target_dir, f'{target}_scaler.pkl')
                    joblib.dump(self.multi_target_models[target]['scaler'], scaler_path)
                
                # Save feature scaler
                feature_scaler_path = os.path.join(target_dir, f'{target}_feature_scaler.pkl')
                joblib.dump(self.scaler, feature_scaler_path)
                
                # Save metrics
                metrics_path = os.path.join(target_dir, f'{target}_metrics.pkl')
                joblib.dump(self.multi_target_models[target]['metrics'], metrics_path)
                
                print(f"Saved {target} model and related files to {target_dir}")
            
            # Save feature columns
            feature_columns_path = os.path.join(self.models_dir, 'feature_columns.pkl')
            joblib.dump(FEATURE_COLUMNS, feature_columns_path)
            
            print("All models saved successfully")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise
    
    def save_model(self):
        """Save the trained models - wrapper for multi-target"""
        if not self.multi_target_models:
            print("Models not trained yet, nothing to save.")
            return
            
        self.save_multi_target_models() 