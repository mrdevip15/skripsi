#!/usr/bin/env python3
"""
Hyperparameter Optimization for Multi-Target GBM Models
Tests 100 different configurations and provides best results with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
import os
from datetime import datetime
import json
import itertools

warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    def __init__(self):
        self.results = []
        self.best_configs = {}
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)',
            'Tavg': 'Average Temperature (°C)',
            'ddd_car': 'Wind Direction (degrees)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        self.output_dir = f"hyperparameter_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
    def create_synthetic_data(self):
        """Create synthetic weather data for demonstration"""
        np.random.seed(42)
        n_samples = 3000
        
        dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
        
        df = pd.DataFrame({
            'Tanggal': dates,
            'Tn': np.random.normal(15, 8, n_samples),
            'Tx': np.random.normal(25, 10, n_samples),
            'RH_avg': np.random.beta(2, 2, n_samples) * 100,
        })
        
        day_of_year = df['Tanggal'].dt.dayofyear
        df['Tn'] += 10 * np.sin(2 * np.pi * day_of_year / 365)
        df['Tx'] += 10 * np.sin(2 * np.pi * day_of_year / 365)
        
        df['Tavg'] = (df['Tn'] + df['Tx']) / 2
        df['RR'] = np.maximum(0, np.random.exponential(5, n_samples) * (df['RH_avg'] / 100))
        df['ss'] = np.maximum(0, 12 - df['RR'] * 0.5 + np.random.normal(0, 2, n_samples))
        df['ff_avg'] = np.maximum(0, np.random.gamma(2, 2, n_samples))
        df['ddd_car'] = np.random.uniform(0, 360, n_samples)
        
        return df
        
    def engineer_features(self, df):
        """Engineer features for the model"""
        df['Temp_Range'] = df['Tx'] - df['Tn']
        df['Dew_Point'] = df['Tavg'] - (100 - df['RH_avg']) / 5
        
        df['Month'] = df['Tanggal'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        df['RR_Rolling_Mean_3d'] = df['RR'].rolling(window=3, min_periods=1).mean()
        df['RR_Rolling_Std_3d'] = df['RR'].rolling(window=3, min_periods=1).std().fillna(0)
        df['RR_Lag_1'] = df['RR'].shift(1).fillna(0)
        df['RR_Lag_2'] = df['RR'].shift(2).fillna(0)
        df['Rain_Binary_Lag_1'] = (df['RR_Lag_1'] > 0).astype(int)
        df['RH_Rolling_Mean_3d'] = df['RH_avg'].rolling(window=3, min_periods=1).mean()
        
        rain_days = (df['RR'] > 0).astype(int)
        df['Rain_Streak'] = rain_days.groupby((rain_days != rain_days.shift()).cumsum()).cumcount() + 1
        df['Rain_Streak'] = df['Rain_Streak'] * rain_days
        
        dry_days = (df['RR'] == 0).astype(int)
        df['Dry_Streak'] = dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumcount() + 1
        df['Dry_Streak'] = df['Dry_Streak'] * dry_days
        
        return df
        
    def generate_configurations(self, n_configs=100):
        """Generate 100 different hyperparameter configurations"""
        n_estimators_range = [50, 100, 150, 200, 250, 300, 400, 500]
        learning_rate_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
        max_depth_range = [3, 4, 5, 6, 7, 8, 9, 10]
        
        all_combinations = list(itertools.product(n_estimators_range, learning_rate_range, max_depth_range))
        
        if len(all_combinations) > n_configs:
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_combinations), n_configs, replace=False)
            configurations = [all_combinations[i] for i in selected_indices]
        else:
            configurations = all_combinations.copy()
            while len(configurations) < n_configs:
                config = (
                    np.random.choice(n_estimators_range),
                    np.random.choice(learning_rate_range),
                    np.random.choice(max_depth_range)
                )
                if config not in configurations:
                    configurations.append(config)
        
        return configurations[:n_configs]
        
    def evaluate_configuration(self, config, X, y):
        """Evaluate a single configuration"""
        n_estimators, learning_rate, max_depth = config
        
        model = GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_val_scaled)
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            scores.append({
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'training_time': training_time
            })
        
        avg_scores = {
            'r2': np.mean([s['r2'] for s in scores]),
            'rmse': np.mean([s['rmse'] for s in scores]),
            'mae': np.mean([s['mae'] for s in scores]),
            'training_time': np.mean([s['training_time'] for s in scores]),
            'r2_std': np.std([s['r2'] for s in scores])
        }
        
        return avg_scores
        
    def optimize_hyperparameters(self, df):
        """Run hyperparameter optimization for all targets"""
        print("Starting hyperparameter optimization...")
        
        feature_columns = [
            'Tn', 'Tx', 'RH_avg', 'Month_sin', 'Month_cos',
            'Temp_Range', 'Dew_Point', 'RR_Rolling_Mean_3d',
            'RR_Rolling_Std_3d', 'RH_Rolling_Mean_3d', 'RR_Lag_1',
            'RR_Lag_2', 'Rain_Binary_Lag_1', 'Rain_Streak', 'Dry_Streak'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Using {len(available_features)} features")
        
        configurations = self.generate_configurations(100)
        
        for target in self.target_columns:
            if target not in df.columns:
                continue
                
            print(f"\nOptimizing for target: {target}")
            
            X = df[available_features].fillna(0).values
            y = df[target].values
            
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            target_results = []
            
            for i, config in enumerate(configurations):
                if i % 20 == 0:
                    print(f"  Progress: {i}/100 configurations tested")
                    
                try:
                    scores = self.evaluate_configuration(config, X, y)
                    
                    result = {
                        'target': target,
                        'config_id': i,
                        'n_estimators': config[0],
                        'learning_rate': config[1],
                        'max_depth': config[2],
                        'r2_score': scores['r2'],
                        'rmse': scores['rmse'],
                        'mae': scores['mae'],
                        'training_time': scores['training_time'],
                        'r2_std': scores['r2_std']
                    }
                    
                    target_results.append(result)
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"Error with config {config}: {e}")
                    continue
            
            if target_results:
                best_config = max(target_results, key=lambda x: x['r2_score'])
                self.best_configs[target] = best_config
                print(f"Best R² for {target}: {best_config['r2_score']:.4f}")
        
        print(f"\nOptimization complete! Total configurations tested: {len(self.results)}")
        
    def create_results_table(self):
        """Create a comprehensive results table"""
        print("Creating results table...")
        
        results_df = pd.DataFrame(self.results)
        
        best_results = []
        for target in self.target_columns:
            if target in self.best_configs:
                best = self.best_configs[target]
                best_results.append(best)
        
        best_df = pd.DataFrame(best_results)
        
        results_df.to_csv(os.path.join(self.output_dir, 'all_results.csv'), index=False)
        best_df.to_csv(os.path.join(self.output_dir, 'best_results.csv'), index=False)
        
        if not best_df.empty:
            display_table = best_df[['target', 'n_estimators', 'learning_rate', 'max_depth', 
                                   'r2_score', 'rmse', 'mae', 'training_time']].copy()
            display_table['target_name'] = display_table['target'].map(self.target_names)
            display_table = display_table.round(4)
            
            print("\n" + "="*80)
            print("BEST HYPERPARAMETER CONFIGURATIONS")
            print("="*80)
            print(display_table[['target_name', 'n_estimators', 'learning_rate', 'max_depth', 'r2_score', 'rmse']].to_string(index=False))
            print("="*80)
            
        return results_df, best_df
        
    def create_accuracy_plots(self, results_df):
        """Create comprehensive accuracy plots"""
        print("Creating accuracy plots...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Best R² for each target
        best_r2_by_target = results_df.groupby('target')['r2_score'].max().reset_index()
        bars = axes[0,0].bar(range(len(best_r2_by_target)), best_r2_by_target['r2_score'])
        axes[0,0].set_xlabel('Target Variable')
        axes[0,0].set_ylabel('Best R² Score')
        axes[0,0].set_title('Best R² Score by Target Variable')
        axes[0,0].set_xticks(range(len(best_r2_by_target)))
        axes[0,0].set_xticklabels([self.target_names.get(t, t) for t in best_r2_by_target['target']], 
                                  rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom')
        
        # 2. R² vs n_estimators
        for target in results_df['target'].unique():
            target_data = results_df[results_df['target'] == target]
            axes[0,1].scatter(target_data['n_estimators'], target_data['r2_score'], 
                             label=self.target_names.get(target, target), alpha=0.6)
        axes[0,1].set_xlabel('Number of Estimators')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_title('R² Score vs Number of Estimators')
        axes[0,1].legend()
        
        # 3. R² vs learning_rate
        for target in results_df['target'].unique():
            target_data = results_df[results_df['target'] == target]
            axes[0,2].scatter(target_data['learning_rate'], target_data['r2_score'], 
                             label=self.target_names.get(target, target), alpha=0.6)
        axes[0,2].set_xlabel('Learning Rate')
        axes[0,2].set_ylabel('R² Score')
        axes[0,2].set_title('R² Score vs Learning Rate')
        axes[0,2].set_xscale('log')
        
        # 4. R² vs max_depth
        for target in results_df['target'].unique():
            target_data = results_df[results_df['target'] == target]
            axes[1,0].scatter(target_data['max_depth'], target_data['r2_score'], 
                             label=self.target_names.get(target, target), alpha=0.6)
        axes[1,0].set_xlabel('Max Depth')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].set_title('R² Score vs Max Depth')
        
        # 5. Training time vs R²
        for target in results_df['target'].unique():
            target_data = results_df[results_df['target'] == target]
            axes[1,1].scatter(target_data['training_time'], target_data['r2_score'], 
                             label=self.target_names.get(target, target), alpha=0.6)
        axes[1,1].set_xlabel('Training Time (seconds)')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('Training Time vs R² Score')
        axes[1,1].set_xscale('log')
        
        # 6. RMSE distribution
        rmse_by_target = []
        target_labels = []
        for target in results_df['target'].unique():
            target_data = results_df[results_df['target'] == target]
            rmse_by_target.append(target_data['rmse'].values)
            target_labels.append(self.target_names.get(target, target))
        
        axes[1,2].boxplot(rmse_by_target, labels=target_labels)
        axes[1,2].set_ylabel('RMSE')
        axes[1,2].set_title('RMSE Distribution by Target')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_plots.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_best_configurations(self):
        """Save best configurations to JSON"""
        config_summary = {}
        for target, config in self.best_configs.items():
            config_summary[target] = {
                'n_estimators': int(config['n_estimators']),
                'learning_rate': float(config['learning_rate']),
                'max_depth': int(config['max_depth']),
                'r2_score': float(config['r2_score']),
                'rmse': float(config['rmse']),
                'mae': float(config['mae']),
                'training_time': float(config['training_time'])
            }
        
        with open(os.path.join(self.output_dir, 'best_configurations.json'), 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        with open(os.path.join(self.output_dir, 'optimization_summary.txt'), 'w') as f:
            f.write("Hyperparameter Optimization Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total configurations tested: {len(self.results)}\n")
            f.write(f"Targets optimized: {len(self.best_configs)}\n\n")
            
            for target, config in self.best_configs.items():
                f.write(f"{self.target_names.get(target, target)}:\n")
                f.write(f"  Best R² Score: {config['r2_score']:.4f}\n")
                f.write(f"  N Estimators: {config['n_estimators']}\n")
                f.write(f"  Learning Rate: {config['learning_rate']}\n")
                f.write(f"  Max Depth: {config['max_depth']}\n")
                f.write(f"  RMSE: {config['rmse']:.4f}\n")
                f.write(f"  Training Time: {config['training_time']:.2f}s\n\n")
    
    def run_optimization(self):
        """Run the complete optimization process"""
        print("Starting Hyperparameter Optimization for Multi-Target GBM Models")
        print("="*70)
        
        # Create synthetic data
        print("Creating synthetic weather data...")
        df = self.create_synthetic_data()
        df = self.engineer_features(df)
        df = df.dropna(subset=self.target_columns)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['Tanggal'].min()} to {df['Tanggal'].max()}")
        
        # Run optimization
        self.optimize_hyperparameters(df)
        
        # Create results
        results_df, best_df = self.create_results_table()
        
        # Create plots
        self.create_accuracy_plots(results_df)
        
        # Save configurations
        self.save_best_configurations()
        
        print(f"\nOptimization complete! Results saved to: {self.output_dir}")
        return self.best_configs, results_df

def main():
    """Main function to run the optimization"""
    optimizer = HyperparameterOptimizer()
    best_configs, results_df = optimizer.run_optimization()
    
    print("\nOptimization completed successfully!")
    print(f"Check the '{optimizer.output_dir}' directory for detailed results and plots.")
    
    return best_configs, results_df

if __name__ == "__main__":
    best_configs, results = main()
