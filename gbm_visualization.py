import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from gbm_config import *

class GBMVisualizer:
    """Handles all visualization tasks for the GBM weather prediction system"""
    
    def __init__(self, plots_dir=PLOTS_DIR):
        self.plots_dir = plots_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.plots_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_target_distributions(self, df, save_path=None, title_prefix="Target Variable Distribution"):
        """Plot distributions of all target variables"""
        if save_path is None:
            save_path = self.run_dir
            
        n_targets = len(TARGET_COLUMNS)
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, target in enumerate(TARGET_COLUMNS, 1):
            plt.subplot(n_rows, n_cols, i)
            if target in df.columns:
                plt.hist(df[target].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title(f'{TARGET_NAMES[target]}')
                plt.xlabel(target)
                plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{title_prefix.lower().replace(' ', '_')}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rainfall_distribution(self, df, save_path=None, title="Rainfall Distribution"):
        """Plot rainfall distribution - updated to use new method"""
        self.plot_target_distributions(df, save_path, title)
    
    def plot_feature_correlations(self, df, save_path=None):
        """Plot correlation matrix of features"""
        if save_path is None:
            save_path = self.run_dir
            
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self, df, save_path=None):
        """Plot distributions of key features"""
        if save_path is None:
            save_path = self.run_dir
            
        # Select key features for visualization
        key_features = ['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg', 'Temp_Range', 'Dew_Point']
        available_features = [f for f in key_features if f in df.columns]
        
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, feature in enumerate(available_features, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.hist(df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_seasonal_patterns(self, df, save_path=None):
        """Plot seasonal patterns in the data"""
        if save_path is None:
            save_path = self.run_dir
            
        # Create monthly averages
        df_copy = df.copy()
        df_copy['Month'] = pd.to_datetime(df_copy['Tanggal']).dt.month
        
        monthly_stats = {}
        for target in TARGET_COLUMNS:
            if target in df_copy.columns:
                monthly_stats[target] = df_copy.groupby('Month')[target].mean()
        
        # Plot monthly patterns
        n_targets = len(monthly_stats)
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4*n_rows))
        for i, (target, monthly_data) in enumerate(monthly_stats.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            monthly_data.plot(kind='line', marker='o', linewidth=2, markersize=6)
            plt.title(f'Monthly {TARGET_NAMES[target]} Pattern')
            plt.xlabel('Month')
            plt.ylabel(TARGET_NAMES[target])
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'seasonal_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_splits(self, df, train_size=0.8, save_path=None):
        """Plot data splits for training and testing"""
        if save_path is None:
            save_path = self.run_dir
            
        # Calculate split point
        split_idx = int(len(df) * train_size)
        
        plt.figure(figsize=(15, 10))
        
        # Plot time series with split
        for i, target in enumerate(TARGET_COLUMNS, 1):
            if target not in df.columns:
                continue
                
            plt.subplot(len(TARGET_COLUMNS), 1, i)
            
            # Plot training data
            train_data = df.iloc[:split_idx]
            plt.plot(train_data['Tanggal'], train_data[target], 
                    label='Training Data', color='blue', alpha=0.7)
            
            # Plot testing data
            test_data = df.iloc[split_idx:]
            plt.plot(test_data['Tanggal'], test_data[target], 
                    label='Testing Data', color='red', alpha=0.7)
            
            plt.title(f'{TARGET_NAMES[target]} - Data Split')
            plt.xlabel('Date')
            plt.ylabel(TARGET_NAMES[target])
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'data_splits.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, dates, actual, predicted, target_name='Target', save_path=None):
        """Plot actual vs predicted values"""
        if save_path is None:
            save_path = self.run_dir
            
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(dates, actual, label='Actual', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predicted values
        plt.plot(dates, predicted, label='Predicted', color='red', linewidth=2, alpha=0.8)
        
        plt.title(f'{target_name} - Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'predictions_{target_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "deg").replace("/", "_")}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multi_target_feature_importance(self, target_models, save_path=None):
        """Plot feature importance for all targets"""
        if save_path is None:
            save_path = self.run_dir
            
        n_targets = len(target_models)
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 6*n_rows))
        
        for i, (target, model_info) in enumerate(target_models.items(), 1):
            if 'model' not in model_info:
                continue
                
            model = model_info['model']
            feature_importance = model.feature_importances_
            
            # Get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            
            # Sort by importance
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_features = feature_names[sorted_idx][:10]  # Top 10 features
            top_importance = feature_importance[sorted_idx][:10]
            
            plt.subplot(n_rows, n_cols, i)
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'{TARGET_NAMES[target]} - Feature Importance')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'multi_target_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_feature_importance(self, target_models, save_path=None):
        """Create comprehensive feature importance visualization"""
        if save_path is None:
            save_path = self.run_dir
            
        # Collect all feature importances
        all_importances = {}
        feature_names = None
        
        for target, model_info in target_models.items():
            if 'model' not in model_info:
                continue
                
            model = model_info['model']
            importance = model.feature_importances_
            
            if feature_names is None:
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            all_importances[target] = importance
        
        if not all_importances:
            print("No models available for feature importance visualization")
            return
        
        # Create heatmap
        importance_df = pd.DataFrame(all_importances, index=feature_names)
        
        plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
        sns.heatmap(importance_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Comprehensive Feature Importance Across All Targets', fontsize=14, pad=20)
        plt.xlabel('Target Variables')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'comprehensive_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, target_models=None, save_path=None):
        """Plot feature importance - wrapper for multi-target version"""
        if target_models is None:
            print("No target models provided for feature importance plotting")
            return
            
        self.plot_multi_target_feature_importance(target_models, save_path)
        self.plot_comprehensive_feature_importance(target_models, save_path)
    
    def plot_predictions_for_day(self, dates, actual, predicted, day, metrics, 
                                title_suffix="", save_path=None):
        """Plot predictions for a specific forecast day"""
        if save_path is None:
            save_path = self.run_dir
            
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(dates, actual, label='Actual', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predicted values
        plt.plot(dates, predicted, label='Predicted', color='red', linewidth=2, alpha=0.8)
        
        # Add metrics to title
        metrics_text = f"RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}"
        title = f'Day {day} Forecast {title_suffix} - {metrics_text}'
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'day_{day}_forecast{title_suffix.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multi_day_predictions(self, results, save_path=None):
        """Plot multi-day prediction results"""
        if save_path is None:
            save_path = self.run_dir
            
        for target, target_results in results.items():
            if target not in TARGET_COLUMNS:
                continue
                
            n_days = len(target_results)
            n_cols = 3
            n_rows = (n_days + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4*n_rows))
            
            for day, day_results in target_results.items():
                if 'dates' not in day_results or 'actual' not in day_results or 'predicted' not in day_results:
                    continue
                    
                plt.subplot(n_rows, n_cols, day)
                
                dates = day_results['dates']
                actual = day_results['actual']
                predicted = day_results['predicted']
                metrics = day_results.get('metrics', {})
                
                plt.plot(dates, actual, label='Actual', color='blue', alpha=0.8)
                plt.plot(dates, predicted, label='Predicted', color='red', alpha=0.8)
                
                title = f'Day {day} - {TARGET_NAMES[target]}'
                if metrics:
                    title += f'\nR²: {metrics.get("R2", 0):.3f}'
                
                plt.title(title)
                plt.xlabel('Date')
                plt.ylabel(TARGET_NAMES[target])
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            filename = f'multi_day_predictions_{target}.png'
            plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_multi_day_summary(self, results, save_path=None):
        """Plot summary of multi-day forecasting performance"""
        if save_path is None:
            save_path = self.run_dir
            
        # Extract metrics for summary
        summary_data = {}
        
        for target, target_results in results.items():
            if target not in TARGET_COLUMNS:
                continue
                
            summary_data[target] = {}
            for day, day_results in target_results.items():
                if 'metrics' in day_results:
                    summary_data[target][day] = day_results['metrics']
        
        # Create performance heatmap
        if summary_data:
            # Prepare data for heatmap
            targets = list(summary_data.keys())
            days = list(range(1, FORECAST_DAYS + 1))
            
            # Create R² heatmap
            r2_matrix = np.zeros((len(targets), len(days)))
            rmse_matrix = np.zeros((len(targets), len(days)))
            
            for i, target in enumerate(targets):
                for j, day in enumerate(days):
                    if day in summary_data[target]:
                        r2_matrix[i, j] = summary_data[target][day].get('R2', 0)
                        rmse_matrix[i, j] = summary_data[target][day].get('RMSE', 0)
            
            # Plot R² heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=[f'Day {d}' for d in days],
                       yticklabels=[TARGET_NAMES[t] for t in targets])
            plt.title('Multi-Day Forecasting Performance (R²)', fontsize=14, pad=20)
            plt.xlabel('Forecast Horizon (Days)')
            plt.ylabel('Target Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multi_day_performance_r2.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot RMSE heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(rmse_matrix, annot=True, fmt='.3f', cmap='RdYlBu',
                       xticklabels=[f'Day {d}' for d in days],
                       yticklabels=[TARGET_NAMES[t] for t in targets])
            plt.title('Multi-Day Forecasting Performance (RMSE)', fontsize=14, pad=20)
            plt.xlabel('Forecast Horizon (Days)')
            plt.ylabel('Target Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multi_day_performance_rmse.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close() 