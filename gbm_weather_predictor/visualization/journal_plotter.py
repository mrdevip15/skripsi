"""
Journal-quality plotting module for GBM Weather Predictor
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ..config.settings import TARGET_NAMES

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class JournalPlotter:
    def __init__(self, output_dir="journal_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Publication-quality colors
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red
            'info': '#3A1772',         # Dark Purple
            'warning': '#F4A261',      # Light Orange
            'light': '#E9C46A',        # Yellow
            'dark': '#264653'          # Dark Blue
        }
        
        # Set font sizes for publication
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

    def create_target_distribution_plot(self, df, save_name="Figure_1_Target_Distributions.png"):
        """Create publication-quality target variable distribution plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribusi Variabel Target Cuaca', fontsize=16, fontweight='bold', y=0.95)
        
        targets = list(TARGET_NAMES.keys())
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['success'], self.colors['info']]
        
        for idx, (target, color) in enumerate(zip(targets, colors)):
            if target in df.columns:
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Create histogram with KDE
                sns.histplot(df[target].dropna(), kde=True, ax=ax, color=color, alpha=0.7)
                ax.set_title(str(TARGET_NAMES[target]), fontweight='bold')
                ax.set_xlabel('Nilai')
                ax.set_ylabel('Frekuensi')
                
                # Add statistics text
                mean_val = df[target].mean()
                std_val = df[target].std()
                ax.text(0.02, 0.98, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplot if needed
        if len(targets) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_correlation_heatmap(self, df, feature_columns, save_name="Figure_2_Feature_Correlation_Matrix.png"):
        """Create publication-quality correlation heatmap"""
        # Select top features and all targets
        if len(feature_columns) > 15:
            # Calculate correlation with targets
            feature_importance = {}
            for feature in feature_columns:
                correlations = []
                for target in TARGET_NAMES.keys():
                    if target in df.columns:
                        corr = abs(np.corrcoef(df[feature], df[target])[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                feature_importance[feature] = np.mean(correlations) if correlations else 0
            
            # Get top 15 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_features = [f[0] for f in top_features]
        else:
            selected_features = feature_columns
        
        # Add target variables
        features_to_plot = selected_features + [t for t in TARGET_NAMES.keys() if t in df.columns]
        features_to_plot = [f for f in features_to_plot if f in df.columns]
        
        # Create correlation matrix
        corr = df[features_to_plot].corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create custom colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, center=0,
                   square=True, cbar_kws={"shrink": .8}, fmt='.2f',
                   linewidths=0.5, linecolor='white')
        
        plt.title('Matriks Korelasi Fitur dengan Variabel Target', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_feature_importance_plot(self, target_models, feature_columns, save_name="Figure_3_Multi_Target_Feature_Importance.png"):
        """Create publication-quality feature importance plot"""
        n_targets = len(target_models)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Kepentingan Fitur untuk Setiap Variabel Target', fontsize=16, fontweight='bold', y=0.95)
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['success'], self.colors['info']]
        
        for idx, (target, model_data) in enumerate(target_models.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get feature importance
            feature_importance = model_data['model'].feature_importances_
            
            # Get top 8 features
            indices = np.argsort(feature_importance)[::-1][:8]
            sorted_feature_names = [feature_columns[idx] for idx in indices]
            sorted_importance = feature_importance[indices]
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(sorted_importance)), sorted_importance, 
                          color=colors[idx % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center', fontsize=9)
            
            ax.set_yticks(range(len(sorted_importance)))
            ax.set_yticklabels(sorted_feature_names, fontsize=9)
            ax.set_xlabel('Kepentingan Fitur')
            ax.set_title(str(TARGET_NAMES[target]), fontweight='bold')
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
        
        # Hide empty subplot if needed
        if n_targets < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_seasonal_patterns_plot(self, df, save_name="Figure_4_Seasonal_Patterns.png"):
        """Create publication-quality seasonal patterns plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pola Musiman Variabel Target', fontsize=16, fontweight='bold', y=0.95)
        
        targets = list(TARGET_NAMES.keys())
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['success'], self.colors['info']]
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                      'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        
        for idx, (target, color) in enumerate(zip(targets, colors)):
            if target in df.columns:
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Calculate monthly averages
                monthly_avg = df.groupby('Month')[target].mean()
                
                # Create bar plot
                bars = ax.bar(range(12), monthly_avg.values, color=color, alpha=0.8)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, monthly_avg.values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.1f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_title(str(TARGET_NAMES[target]), fontweight='bold')
                ax.set_xlabel('Bulan')
                ax.set_ylabel('Rata-rata')
                ax.set_xticks(range(12))
                ax.set_xticklabels(month_names, rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplot if needed
        if len(targets) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_prediction_performance_plot(self, target_models, save_name="Figure_5_Prediction_Performance.png"):
        """Create publication-quality prediction performance comparison"""
        targets = list(target_models.keys())
        metrics = ['R2', 'RMSE', 'MAE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performa Prediksi Model GBM', fontsize=16, fontweight='bold', y=0.95)
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['success'], self.colors['info']]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract metric values
            values = []
            labels = []
            for target in targets:
                if metric in target_models[target]['metrics']:
                    values.append(target_models[target]['metrics'][metric])
                    labels.append(TARGET_NAMES[target])
            
            # Create bar plot
            bars = ax.bar(range(len(values)), values, 
                         color=colors[:len(values)], alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{metric} Score', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits for better visualization
            if metric == 'R2':
                ax.set_ylim(0, 1)
            elif metric in ['RMSE', 'MAE']:
                ax.set_ylim(0, max(values) * 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_actual_vs_predicted_plot(self, dates, actual, predicted, target_name, save_name=None):
        """Create publication-quality actual vs predicted plot"""
        if save_name is None:
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
                               .replace('°', '')
                               .replace('(', '')
                               .replace(')', '')
                               .replace('m/s', 'ms')
                               .lower())
            save_name = f"Figure_6_{target_safe_name}_Predictions.png"
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Aktual vs Prediksi {target_name}', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Time series
        ax1.plot(dates, actual, marker='o', linestyle='-', label='Aktual', 
                color=self.colors['primary'], alpha=0.7, markersize=3)
        ax1.plot(dates, predicted, marker='x', linestyle='-', label='Prediksi', 
                color=self.colors['secondary'], alpha=0.7, markersize=3)
        ax1.set_title('Seri Waktu Aktual vs Prediksi', fontweight='bold')
        ax1.set_xlabel('Tanggal')
        ax1.set_ylabel(target_name)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Scatter plot
        ax2.scatter(actual, predicted, alpha=0.6, color=self.colors['accent'])
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_title('Scatter Plot Aktual vs Prediksi', fontweight='bold')
        ax2.set_xlabel('Nilai Aktual')
        ax2.set_ylabel('Nilai Prediksi')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add R² value
        r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created {save_name}")

    def create_comprehensive_journal_plots(self, df, target_models, feature_columns, test_data=None):
        """Create all journal-quality plots"""
        print("\n🎨 Creating Journal-Quality Plots...")
        
        # Create all plots
        self.create_target_distribution_plot(df)
        self.create_correlation_heatmap(df, feature_columns)
        self.create_feature_importance_plot(target_models, feature_columns)
        self.create_seasonal_patterns_plot(df)
        self.create_prediction_performance_plot(target_models)
        
        # Create actual vs predicted plots if test data is provided
        if test_data:
            for target in target_models.keys():
                if target in test_data['actual'].columns and target in test_data['predicted'].columns:
                    self.create_actual_vs_predicted_plot(
                        test_data['dates'],
                        test_data['actual'][target],
                        test_data['predicted'][target],
                        TARGET_NAMES[target]
                    )
        
        print(f"\n✅ All journal plots created in {self.output_dir}/")
        print("📊 Ready for publication!") 