#!/usr/bin/env python3
"""
Optimized Hyperparameter Search with Early Stopping
- N Estimators: 1 to 400 (stop if R² > 0.95)
- Max Depth: 1 to 5
- Learning Rate: 0.010 to 0.150
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

warnings.filterwarnings('ignore')

class OptimizedHyperparameterSearch:
    def __init__(self):
        self.results = []
        self.best_configs = {}
        self.early_stopped_configs = {}
        self.target_columns = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ddd_car': 'Wind Direction (degrees)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        self.output_dir = f"optimized_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        
    def evaluate_configuration(self, n_estimators, learning_rate, max_depth, X, y):
        """Evaluate a single configuration"""
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
        
    def optimized_search_for_target(self, target, X, y):
        """Optimized search for a single target with early stopping"""
        print(f"\n{'='*60}")
        print(f"�� OPTIMIZING: {self.target_names[target]}")
        print(f"{'='*60}")
        
        best_config = None
        best_r2 = -float('inf')
        all_results = []
        early_stop_info = {
            'early_stopped': False,
            'stop_reason': '',
            'configs_tested': 0
        }
        
        # Parameter ranges as specified
        learning_rates = np.arange(0.010, 0.151, 0.005)  # 0.010 to 0.150 in steps of 0.005
        max_depths = range(1, 6)  # 1 to 5
        n_estimators_range = range(1, 401)  # 1 to 400
        
        total_possible_configs = len(learning_rates) * len(max_depths) * len(n_estimators_range)
        print(f"📊 Total possible configurations: {total_possible_configs:,}")
        print(f"🎯 Target accuracy for early stopping: >95% (R² > 0.95)")
        
        configs_tested = 0
        
        # Nested loops with early stopping
        for lr in learning_rates:
            for max_depth in max_depths:
                for n_est in n_estimators_range:
                    configs_tested += 1
                    
                    try:
                        scores = self.evaluate_configuration(n_est, lr, max_depth, X, y)
                        
                        result = {
                            'target': target,
                            'n_estimators': n_est,
                            'learning_rate': lr,
                            'max_depth': max_depth,
                            'r2_score': scores['r2'],
                            'rmse': scores['rmse'],
                            'mae': scores['mae'],
                            'training_time': scores['training_time'],
                            'r2_std': scores['r2_std']
                        }
                        
                        all_results.append(result)
                        self.results.append(result)
                        
                        # Update best configuration
                        if scores['r2'] > best_r2:
                            best_r2 = scores['r2']
                            best_config = result.copy()
                            
                        # Progress reporting
                        if configs_tested % 50 == 0 or scores['r2'] > 0.9:
                            print(f"  📈 Progress: {configs_tested:4d} configs | "
                                  f"Current R²: {scores['r2']:.4f} | "
                                  f"Best R²: {best_r2:.4f} | "
                                  f"Config: n_est={n_est}, lr={lr:.3f}, depth={max_depth}")
                        
                        # Early stopping check - if R² > 95%
                        if scores['r2'] > 0.95:
                            early_stop_info['early_stopped'] = True
                            early_stop_info['stop_reason'] = f"Achieved {scores['r2']:.4f} R² (>95%)"
                            early_stop_info['configs_tested'] = configs_tested
                            
                            print(f"  🎉 EARLY STOP! Achieved {scores['r2']:.4f} R² (>95%)")
                            print(f"  ⚡ Optimal config found: n_est={n_est}, lr={lr:.3f}, depth={max_depth}")
                            break
                            
                    except Exception as e:
                        print(f"  ❌ Error with config n_est={n_est}, lr={lr:.3f}, depth={max_depth}: {e}")
                        continue
                
                if early_stop_info['early_stopped']:
                    break
            if early_stop_info['early_stopped']:
                break
        
        if not early_stop_info['early_stopped']:
            early_stop_info['configs_tested'] = configs_tested
            early_stop_info['stop_reason'] = "Completed full search"
        
        print(f"\n📊 SEARCH SUMMARY for {self.target_names[target]}:")
        print(f"  ✅ Configurations tested: {early_stop_info['configs_tested']:,}")
        print(f"  🏆 Best R² achieved: {best_r2:.4f} ({best_r2*100:.2f}%)")
        print(f"  ⏱️  Stop reason: {early_stop_info['stop_reason']}")
        
        if best_config:
            self.best_configs[target] = best_config
            self.early_stopped_configs[target] = early_stop_info
            
        return all_results, early_stop_info
        
    def run_optimized_search(self, df):
        """Run optimized search for all targets"""
        print("🚀 STARTING OPTIMIZED HYPERPARAMETER SEARCH")
        print("="*70)
        
        feature_columns = [
            'Tn', 'Tx', 'RH_avg', 'Month_sin', 'Month_cos',
            'Temp_Range', 'Dew_Point', 'RR_Rolling_Mean_3d',
            'RR_Rolling_Std_3d', 'RH_Rolling_Mean_3d', 'RR_Lag_1',
            'RR_Lag_2', 'Rain_Binary_Lag_1', 'Rain_Streak', 'Dry_Streak'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"📊 Using {len(available_features)} features")
        
        total_start_time = time.time()
        
        for target in self.target_columns:
            if target not in df.columns:
                continue
                
            X = df[available_features].fillna(0).values
            y = df[target].values
            
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            target_results, early_stop_info = self.optimized_search_for_target(target, X, y)
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Total configurations tested: {len(self.results)}")
        
    def create_results_summary(self):
        """Create comprehensive results summary"""
        print("\n📊 CREATING RESULTS SUMMARY...")
        
        results_df = pd.DataFrame(self.results)
        
        # Best results summary
        best_results = []
        for target in self.target_columns:
            if target in self.best_configs:
                best = self.best_configs[target].copy()
                best['early_stopped'] = self.early_stopped_configs[target]['early_stopped']
                best['configs_tested'] = self.early_stopped_configs[target]['configs_tested']
                best['stop_reason'] = self.early_stopped_configs[target]['stop_reason']
                best_results.append(best)
        
        best_df = pd.DataFrame(best_results)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'all_results.csv'), index=False)
        best_df.to_csv(os.path.join(self.output_dir, 'best_results.csv'), index=False)
        
        # Display results table
        if not best_df.empty:
            print("\n" + "="*120)
            print(" " * 40 + "🏆 OPTIMIZED SEARCH RESULTS")
            print("="*120)
            
            print(f"{'Rank':<4} {'Target Variable':<25} {'R² Score':<9} {'N Est':<6} {'LR':<6} {'Depth':<5} "
                  f"{'RMSE':<8} {'Configs':<8} {'Early Stop':<10}")
            print("-" * 120)
            
            # Sort by R² score
            sorted_results = best_df.sort_values('r2_score', ascending=False)
            
            for rank, (_, row) in enumerate(sorted_results.iterrows(), 1):
                early_stop = "YES" if row['early_stopped'] else "NO"
                print(f"{rank:<4} {self.target_names[row['target']]:<25} "
                      f"{row['r2_score']:<9.4f} {row['n_estimators']:<6} "
                      f"{row['learning_rate']:<6.3f} {row['max_depth']:<5} "
                      f"{row['rmse']:<8.3f} {row['configs_tested']:<8} {early_stop:<10}")
            
            print("-" * 120)
            
            # Efficiency analysis
            print(f"\n📈 EFFICIENCY ANALYSIS:")
            total_configs = sum(self.early_stopped_configs[t]['configs_tested'] for t in self.early_stopped_configs)
            early_stops = sum(1 for t in self.early_stopped_configs if self.early_stopped_configs[t]['early_stopped'])
            
            print(f"  • Total configurations tested: {total_configs:,}")
            print(f"  • Targets with early stopping: {early_stops}/{len(self.early_stopped_configs)}")
            print(f"  • Average configs per target: {total_configs/len(self.early_stopped_configs):.0f}")
            
            # Best performers
            best_target = sorted_results.iloc[0]
            print(f"\n🏆 CHAMPION: {self.target_names[best_target['target']]}")
            print(f"  • R² Score: {best_target['r2_score']:.4f} ({best_target['r2_score']*100:.2f}%)")
            print(f"  • Configuration: n_estimators={best_target['n_estimators']}, "
                  f"lr={best_target['learning_rate']:.3f}, depth={best_target['max_depth']}")
            print(f"  • Early stopped: {best_target['early_stopped']}")
            print(f"  • Configs tested: {best_target['configs_tested']}")
            
        return results_df, best_df
        
    def save_configurations(self):
        """Save best configurations and summary"""
        config_summary = {}
        for target, config in self.best_configs.items():
            config_summary[target] = {
                'n_estimators': int(config['n_estimators']),
                'learning_rate': float(config['learning_rate']),
                'max_depth': int(config['max_depth']),
                'r2_score': float(config['r2_score']),
                'rmse': float(config['rmse']),
                'mae': float(config['mae']),
                'training_time': float(config['training_time']),
                'early_stopped': self.early_stopped_configs[target]['early_stopped'],
                'configs_tested': self.early_stopped_configs[target]['configs_tested'],
                'stop_reason': self.early_stopped_configs[target]['stop_reason']
            }
        
        with open(os.path.join(self.output_dir, 'best_configurations.json'), 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        # Summary report
        with open(os.path.join(self.output_dir, 'optimization_summary.txt'), 'w') as f:
            f.write("Optimized Hyperparameter Search Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Search Parameters:\n")
            f.write(f"• N Estimators: 1 to 400 (early stop if R² > 0.95)\n")
            f.write(f"• Learning Rate: 0.010 to 0.150 (step 0.005)\n") 
            f.write(f"• Max Depth: 1 to 5\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total configurations tested: {len(self.results)}\n\n")
            
            for target, config in self.best_configs.items():
                f.write(f"{self.target_names[target]}:\n")
                f.write(f"  Best R² Score: {config['r2_score']:.4f}\n")
                f.write(f"  Configuration: n_est={config['n_estimators']}, lr={config['learning_rate']:.3f}, depth={config['max_depth']}\n")
                f.write(f"  Early Stopped: {self.early_stopped_configs[target]['early_stopped']}\n")
                f.write(f"  Configs Tested: {self.early_stopped_configs[target]['configs_tested']}\n")
                f.write(f"  RMSE: {config['rmse']:.4f}\n\n")
    
    def run_complete_optimization(self):
        """Run the complete optimization process"""
        print("🚀 OPTIMIZED HYPERPARAMETER SEARCH WITH EARLY STOPPING")
        print("="*70)
        
        # Create data
        print("📊 Creating synthetic weather data...")
        df = self.create_synthetic_data()
        df = self.engineer_features(df)
        df = df.dropna(subset=self.target_columns)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['Tanggal'].min()} to {df['Tanggal'].max()}")
        
        # Run optimization
        self.run_optimized_search(df)
        
        # Create results
        results_df, best_df = self.create_results_summary()
        
        # Save configurations
        self.save_configurations()
        
        print(f"\n🎉 OPTIMIZATION COMPLETE! Results saved to: {self.output_dir}")
        return self.best_configs, results_df

def main():
    """Main function"""
    optimizer = OptimizedHyperparameterSearch()
    best_configs, results_df = optimizer.run_complete_optimization()
    
    print("\n" + "="*70)
    print("✅ OPTIMIZED SEARCH COMPLETED SUCCESSFULLY!")
    print(f"📁 Check the '{optimizer.output_dir}' directory for detailed results.")
    print("="*70)
    
    return best_configs, results_df

if __name__ == "__main__":
    best_configs, results = main()
