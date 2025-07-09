"""
Visualization module for GBM Weather Predictor
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ..config.settings import TARGET_NAMES

class WeatherPlotter:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        
    def plot_target_distributions(self, df, save_path, title_prefix="Distribusi Variabel Target"):
        """Plot distributions of all target variables"""
        n_targets = len([t for t in TARGET_NAMES.keys() if t in df.columns])
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, target in enumerate(TARGET_NAMES.keys(), 1):
            if target in df.columns:
                plt.subplot(n_rows, n_cols, i)
                plt.hist(df[target].dropna(), bins=50, alpha=0.7)
                plt.title(f'{TARGET_NAMES[target]}')
                plt.xlabel(target)
                plt.ylabel('Frekuensi')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{title_prefix.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_feature_correlations(self, df, feature_columns):
        """Plot correlation matrix of features with all target variables"""
        # Use only a subset of features if there are too many
        if len(feature_columns) > 12:  # Reduced to make room for multiple targets
            # Calculate correlation with all targets
            feature_importance = {}
            for feature in feature_columns:
                # Calculate average absolute correlation with all targets
                correlations = []
                for target in TARGET_NAMES.keys():
                    if target in df.columns:
                        corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                feature_importance[feature] = np.mean(correlations) if correlations else 0
            
            # Get the top 12 most correlated features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
            selected_features = [f[0] for f in top_features]
            features_to_plot = selected_features + list(TARGET_NAMES.keys())
            print(f"\nPlotting correlations for top 12 features: {', '.join(selected_features)}")
        else:
            features_to_plot = feature_columns + list(TARGET_NAMES.keys())
        
        # Filter features that exist in the dataframe
        features_to_plot = [f for f in features_to_plot if f in df.columns]
        
        # Create correlation matrix
        corr = df[features_to_plot].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Korelasi Fitur dengan Variabel Target')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_correlations.png'))
        plt.close()

    def plot_feature_distributions(self, df, feature_columns):
        """Plot distribution of top features and all target variables"""
        # Select top features by average correlation with all targets
        feature_importance = {}
        for feature in feature_columns:
            correlations = []
            for target in TARGET_NAMES.keys():
                if target in df.columns:
                    corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
            feature_importance[feature] = np.mean(correlations) if correlations else 0
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:6]
        selected_features = [f[0] for f in top_features]
        
        # Combine top features with all target variables
        features_to_plot = selected_features + [t for t in TARGET_NAMES.keys() if t in df.columns]
        
        # Plot
        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        for i, feature in enumerate(features_to_plot, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[feature], kde=True)
            
            # Add correlation info for features, target name for targets
            if feature in TARGET_NAMES.keys():
                plt.title(f'{TARGET_NAMES.get(feature, feature)} (Target)')
            else:
                avg_corr = feature_importance.get(feature, 0)
                plt.title(f'{feature} (korelasi rata-rata: {avg_corr:.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'feature_distributions.png'))
        plt.close()

    def plot_seasonal_patterns(self, df):
        """Plot seasonal patterns of all target variables"""
        n_targets = len([t for t in TARGET_NAMES.keys() if t in df.columns])
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 4*n_rows))
        plot_idx = 1
        
        for target in TARGET_NAMES.keys():
            if target in df.columns:
                plt.subplot(n_rows, n_cols, plot_idx)
                monthly_avg = df.groupby('Month')[target].mean()
                monthly_avg.plot(kind='bar')
                plt.title(f'Rata-rata {TARGET_NAMES[target]} per Bulan')
                plt.xlabel('Bulan')
                plt.ylabel(TARGET_NAMES[target])
                plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                                     'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'], rotation=45)
                plot_idx += 1
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'seasonal_patterns.png'))
        plt.close()

    def plot_predictions(self, dates, actual, predicted, target_name='Target'):
        """Plot actual vs predicted values for a specific target"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, actual, marker='o', linestyle='-', label='Aktual', alpha=0.7)
        plt.plot(dates, predicted, marker='x', linestyle='-', label='Prediksi', alpha=0.7)
        plt.title(f'Aktual vs Prediksi {target_name}')
        plt.xlabel('Tanggal')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Improved filename sanitization - remove all problematic characters
        target_safe_name = (target_name.replace(' ', '_')
                           .replace('(', '')
                           .replace(')', '')
                           .replace('/', '_')
                           .replace('\\', '_')
                           .replace(':', '_')
                           .replace('*', '_')
                           .replace('?', '_')
                           .replace('"', '_')
                           .replace('<', '_')
                           .replace('>', '_')
                           .replace('|', '_')
                           .lower())
        
        plt.savefig(os.path.join(self.run_dir, f'predictions_{target_safe_name}.png'))
        plt.close()

    def plot_multi_target_feature_importance(self, target_models, feature_columns):
        """Plot feature importance for all target variables"""
        n_targets = len(target_models)
        n_cols = 2
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        
        for i, (target, model_data) in enumerate(target_models.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Get feature importance
            feature_importance = model_data['model'].feature_importances_
            
            # Sort features by importance (show top 5)
            indices = np.argsort(feature_importance)[::-1][:5]
            sorted_feature_names = [feature_columns[idx] for idx in indices]
            sorted_importance = feature_importance[indices]
            
            # Plot horizontal bar chart
            plt.barh(range(len(sorted_importance)), sorted_importance)
            plt.yticks(range(len(sorted_importance)), sorted_feature_names)
            plt.xlabel('Kepentingan')
            plt.title(f'5 Fitur Teratas - {TARGET_NAMES[target]}')
            plt.gca().invert_yaxis()  # Highest importance at top
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'multi_target_feature_importance.png'))
        plt.close() 