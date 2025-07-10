import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from gbm_config import *

class GBMOutputGenerator:
    """Generates CSV output tables for feature importance, predictions, and metrics"""
    
    def __init__(self, output_dir=None):
        """Initialize output generator with specified directory"""
        if output_dir is None:
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(GBM_DIR, 'csv_outputs', self.timestamp)
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"CSV outputs will be saved to: {self.output_dir}")
    
    def generate_feature_importance_table(self, target_models, save_path=None):
        """Generate comprehensive feature importance table for all targets"""
        try:
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'feature_importance_all_targets.csv')
            
            # Collect all feature importances
            importance_data = {}
            feature_names = None
            
            for target, model_info in target_models.items():
                if 'model' not in model_info:
                    continue
                    
                model = model_info['model']
                importance = model.feature_importances_
                
                # Get feature names
                if feature_names is None:
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = model.feature_names_in_
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(importance))]
                
                importance_data[f'{target}_importance'] = importance
                
                # Calculate percentage importance
                importance_pct = (importance / importance.sum()) * 100
                importance_data[f'{target}_importance_pct'] = importance_pct
                
                # Calculate rank (1 = most important)
                importance_rank = len(importance) + 1 - np.argsort(np.argsort(importance)) - 1
                importance_data[f'{target}_rank'] = importance_rank
            
            # Create DataFrame
            importance_df = pd.DataFrame(importance_data, index=feature_names)
            importance_df.index.name = 'Feature'
            
            # Add overall importance metrics
            importance_cols = [col for col in importance_df.columns if col.endswith('_importance')]
            importance_df['Average_Importance'] = importance_df[importance_cols].mean(axis=1)
            importance_df['Average_Importance_Pct'] = (importance_df['Average_Importance'] / importance_df['Average_Importance'].sum()) * 100
            importance_df['Overall_Rank'] = len(importance_df) + 1 - np.argsort(np.argsort(importance_df['Average_Importance'])) - 1
            
            # Sort by overall importance
            importance_df = importance_df.sort_values('Average_Importance', ascending=False)
            
            # Round numerical values
            importance_df = importance_df.round(6)
            
            # Save to CSV
            importance_df.to_csv(save_path)
            print(f"Feature importance table saved to: {save_path}")
            
            # Create summary table (top 10 features)
            summary_path = os.path.join(self.output_dir, 'feature_importance_top10.csv')
            top10_df = importance_df.head(10)[['Average_Importance', 'Average_Importance_Pct', 'Overall_Rank']]
            
            # Add target-specific ranks for top 10
            for target in TARGET_COLUMNS:
                rank_col = f'{target}_rank'
                if rank_col in importance_df.columns:
                    top10_df[f'{target}_Rank'] = importance_df[rank_col].head(10)
            
            top10_df.to_csv(summary_path)
            print(f"Top 10 feature importance summary saved to: {summary_path}")
            
            return importance_df
            
        except Exception as e:
            print(f"Error generating feature importance table: {str(e)}")
            raise
    
    def generate_prediction_table(self, df, target_models, scaler, n_samples=10, save_path=None):
        """Generate prediction table for first n samples with dates"""
        try:
            if save_path is None:
                save_path = os.path.join(self.output_dir, f'predictions_first_{n_samples}_samples.csv')
            
            # Prepare features
            available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            X = df[available_features].head(n_samples)
            X_scaled = scaler.transform(X)
            
            # Create results dataframe
            results_data = {}
            
            # Add date information
            if 'Tanggal' in df.columns:
                results_data['Date'] = df['Tanggal'].head(n_samples).values
            else:
                results_data['Index'] = range(n_samples)
            
            # Add sample number
            results_data['Sample_No'] = range(1, n_samples + 1)
            
            # Generate predictions for each target
            for target in TARGET_COLUMNS:
                if target not in target_models or target not in df.columns:
                    continue
                    
                model_info = target_models[target]
                model = model_info['model']
                target_scaler = model_info['scaler']
                
                # Get actual values
                actual_values = df[target].head(n_samples).values
                results_data[f'{target}_Actual'] = actual_values
                
                # Make predictions
                y_pred_scaled = model.predict(X_scaled)
                
                # Inverse transform if scaling was applied
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                else:
                    y_pred = y_pred_scaled
                
                # Apply constraints
                y_pred = self._apply_target_constraints(y_pred, target)
                results_data[f'{target}_Predicted'] = y_pred
                
                # Calculate individual errors
                absolute_error = np.abs(actual_values - y_pred)
                squared_error = (actual_values - y_pred) ** 2
                percentage_error = np.where(actual_values != 0, 
                                          (absolute_error / np.abs(actual_values)) * 100, 
                                          np.nan)
                
                results_data[f'{target}_Absolute_Error'] = absolute_error
                results_data[f'{target}_Squared_Error'] = squared_error
                results_data[f'{target}_Percentage_Error'] = percentage_error
            
            # Create DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Round numerical values
            numerical_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numerical_cols] = results_df[numerical_cols].round(4)
            
            # Save to CSV
            results_df.to_csv(save_path, index=False)
            print(f"Prediction table for first {n_samples} samples saved to: {save_path}")
            
            return results_df
            
        except Exception as e:
            print(f"Error generating prediction table: {str(e)}")
            raise
    
    def generate_metrics_table(self, target_models, multi_day_results=None, save_path=None):
        """Generate comprehensive accuracy metrics table for all models"""
        try:
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'model_accuracy_metrics.csv')
            
            metrics_data = []
            
            # Single-day model metrics
            for target, model_info in target_models.items():
                if 'metrics' not in model_info:
                    continue
                    
                metrics = model_info['metrics']
                
                row = {
                    'Target_Variable': target,
                    'Target_Name': TARGET_NAMES.get(target, target),
                    'Model_Type': 'Single_Day',
                    'Forecast_Day': 1,
                    'RMSE': metrics.get('RMSE', np.nan),
                    'MAE': metrics.get('MAE', np.nan),
                    'MSE': metrics.get('MSE', np.nan),
                    'R2_Score': metrics.get('R2', np.nan)
                }
                
                # Add target-specific metrics
                if target == 'RR' and 'Rain_Detection_Accuracy' in metrics:
                    row['Rain_Detection_Accuracy'] = metrics['Rain_Detection_Accuracy']
                    row['Actual_Rain_Days'] = metrics.get('Actual_Rain_Days', np.nan)
                    row['Predicted_Rain_Days'] = metrics.get('Predicted_Rain_Days', np.nan)
                
                metrics_data.append(row)
            
            # Multi-day model metrics
            if multi_day_results:
                for target, target_results in multi_day_results.items():
                    if target not in TARGET_COLUMNS:
                        continue
                        
                    for day, day_results in target_results.items():
                        if 'metrics' not in day_results:
                            continue
                            
                        metrics = day_results['metrics']
                        
                        row = {
                            'Target_Variable': target,
                            'Target_Name': TARGET_NAMES.get(target, target),
                            'Model_Type': 'Multi_Day',
                            'Forecast_Day': day,
                            'RMSE': metrics.get('RMSE', np.nan),
                            'MAE': metrics.get('MAE', np.nan),
                            'MSE': metrics.get('MSE', np.nan),
                            'R2_Score': metrics.get('R2', np.nan)
                        }
                        
                        # Add target-specific metrics
                        if target == 'RR' and 'Rain_Detection_Accuracy' in metrics:
                            row['Rain_Detection_Accuracy'] = metrics['Rain_Detection_Accuracy']
                            row['Actual_Rain_Days'] = metrics.get('Actual_Rain_Days', np.nan)
                            row['Predicted_Rain_Days'] = metrics.get('Predicted_Rain_Days', np.nan)
                        
                        metrics_data.append(row)
            
            # Create DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            # Sort by target and forecast day
            metrics_df = metrics_df.sort_values(['Target_Variable', 'Forecast_Day'])
            
            # Round numerical values
            numerical_cols = ['RMSE', 'MAE', 'MSE', 'R2_Score', 'Rain_Detection_Accuracy']
            for col in numerical_cols:
                if col in metrics_df.columns:
                    metrics_df[col] = metrics_df[col].round(6)
            
            # Save to CSV
            metrics_df.to_csv(save_path, index=False)
            print(f"Model accuracy metrics table saved to: {save_path}")
            
            # Create summary table
            summary_path = os.path.join(self.output_dir, 'model_performance_summary.csv')
            
            # Calculate summary statistics for single-day models
            single_day_df = metrics_df[metrics_df['Model_Type'] == 'Single_Day']
            summary_data = []
            
            for target in TARGET_COLUMNS:
                target_data = single_day_df[single_day_df['Target_Variable'] == target]
                if len(target_data) > 0:
                    row = target_data.iloc[0].copy()
                    row['Performance_Category'] = self._categorize_performance(row['R2_Score'])
                    summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            if len(summary_df) > 0:
                summary_df = summary_df[['Target_Variable', 'Target_Name', 'RMSE', 'MAE', 'R2_Score', 'Performance_Category']]
                summary_df.to_csv(summary_path, index=False)
                print(f"Model performance summary saved to: {summary_path}")
            
            return metrics_df
            
        except Exception as e:
            print(f"Error generating metrics table: {str(e)}")
            raise
    
    def generate_all_tables(self, df, target_models, scaler, multi_day_results=None, n_prediction_samples=10):
        """Generate all CSV tables in one call"""
        try:
            print("Generating all CSV output tables...")
            
            # 1. Feature importance table
            print("\n1. Generating feature importance table...")
            importance_df = self.generate_feature_importance_table(target_models)
            
            # 2. Prediction table
            print("\n2. Generating prediction table...")
            prediction_df = self.generate_prediction_table(df, target_models, scaler, n_prediction_samples)
            
            # 3. Metrics table
            print("\n3. Generating metrics table...")
            metrics_df = self.generate_metrics_table(target_models, multi_day_results)
            
            # 4. Generate combined summary
            print("\n4. Generating combined summary...")
            self._generate_combined_summary(importance_df, prediction_df, metrics_df)
            
            print(f"\nAll CSV tables generated successfully in: {self.output_dir}")
            
            return {
                'feature_importance': importance_df,
                'predictions': prediction_df,
                'metrics': metrics_df,
                'output_dir': self.output_dir
            }
            
        except Exception as e:
            print(f"Error generating all tables: {str(e)}")
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
    
    def _categorize_performance(self, r2_score):
        """Categorize model performance based on R² score"""
        if pd.isna(r2_score):
            return 'Unknown'
        elif r2_score >= 0.8:
            return 'Excellent'
        elif r2_score >= 0.7:
            return 'Good'
        elif r2_score >= 0.6:
            return 'Moderate'
        elif r2_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_combined_summary(self, importance_df, prediction_df, metrics_df):
        """Generate a combined summary of all results"""
        try:
            summary_path = os.path.join(self.output_dir, 'combined_summary.txt')
            
            with open(summary_path, 'w') as f:
                f.write("GBM WEATHER PREDICTION SYSTEM - CSV OUTPUT SUMMARY\n")
                f.write("=" * 55 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {self.output_dir}\n\n")
                
                # Feature importance summary
                f.write("TOP 5 MOST IMPORTANT FEATURES\n")
                f.write("-" * 30 + "\n")
                top5_features = importance_df.head(5)
                for i, (feature, row) in enumerate(top5_features.iterrows(), 1):
                    f.write(f"{i}. {feature}: {row['Average_Importance_Pct']:.2f}%\n")
                
                f.write("\n")
                
                # Model performance summary
                f.write("MODEL PERFORMANCE SUMMARY\n")
                f.write("-" * 25 + "\n")
                single_day_metrics = metrics_df[metrics_df['Model_Type'] == 'Single_Day']
                for _, row in single_day_metrics.iterrows():
                    f.write(f"{row['Target_Name']:25} - R²: {row['R2_Score']:.4f}, RMSE: {row['RMSE']:.4f}\n")
                
                f.write("\n")
                
                # Prediction accuracy summary
                f.write("PREDICTION ACCURACY SAMPLE\n")
                f.write("-" * 25 + "\n")
                for target in TARGET_COLUMNS:
                    actual_col = f'{target}_Actual'
                    pred_col = f'{target}_Predicted'
                    error_col = f'{target}_Percentage_Error'
                    
                    if all(col in prediction_df.columns for col in [actual_col, pred_col, error_col]):
                        avg_error = prediction_df[error_col].mean()
                        f.write(f"{TARGET_NAMES.get(target, target):25} - Avg Error: {avg_error:.2f}%\n")
                
                f.write("\n")
                f.write("FILES GENERATED:\n")
                f.write("-" * 15 + "\n")
                f.write("1. feature_importance_all_targets.csv\n")
                f.write("2. feature_importance_top10.csv\n")
                f.write("3. predictions_first_10_samples.csv\n")
                f.write("4. model_accuracy_metrics.csv\n")
                f.write("5. model_performance_summary.csv\n")
                f.write("6. combined_summary.txt\n")
            
            print(f"Combined summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"Error generating combined summary: {str(e)}") 