#!/usr/bin/env python3
"""
Smart Hyperparameter Search with Early Stopping
More efficient strategy: Test parameters in order of importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

class SmartHyperparameterSearch:
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
        
        self.output_dir = f"smart_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
    def create_synthetic_data(self):
        """Create synthetic weather data"""
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
        """Engineer features"""
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
        
    def evaluate_config_fast(self, n_estimators, learning_rate, max_depth, X, y):
        """Fast evaluation using single split"""
        split_point = int(0.8 * len(X))
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
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
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_val_scaled)
        
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time
        }
        
    def smart_search_for_target(self, target, X, y):
        """Smart search strategy - test most promising configurations first"""
        print(f"\n{'='*60}")
        print(f"🎯 SMART SEARCH: {self.target_names[target]}")
        print(f"{'='*60}")
        
        best_config = None
        best_r2 = -float('inf')
        configs_tested = 0
        
        # Smart parameter ranges - test promising values first
        learning_rates = [0.1, 0.15, 0.08, 0.05, 0.12, 0.2, 0.03, 0.01]  # Start with best from previous
        max_depths = [3, 4, 5, 2, 1]  # Start with depth 3 (best from previous)
        
        print(f"🧠 Smart strategy: Testing promising parameters first")
        print(f"🎯 Target: >95% accuracy (R² > 0.95)")
        
        # Phase 1: Quick scan with fewer n_estimators
        print(f"\n📊 PHASE 1: Quick parameter scan")
        n_estimators_quick = [50, 100, 200, 300]
        
        for lr in learning_rates:
            if lr < 0.010 or lr > 0.150:  # Respect user's constraints
                continue
                
            for depth in max_depths:
                for n_est in n_estimators_quick:
                    configs_tested += 1
                    
                    try:
                        scores = self.evaluate_config_fast(n_est, lr, depth, X, y)
                        
                        result = {
                            'target': target,
                            'n_estimators': n_est,
                            'learning_rate': lr,
                            'max_depth': depth,
                            'r2_score': scores['r2'],
                            'rmse': scores['rmse'],
                            'mae': scores['mae'],
                            'training_time': scores['training_time']
                        }
                        
                        self.results.append(result)
                        
                        if scores['r2'] > best_r2:
                            best_r2 = scores['r2']
                            best_config = result.copy()
                            
                        print(f"  Config {configs_tested:3d}: n_est={n_est:3d}, lr={lr:.3f}, depth={depth} "
                              f"-> R²={scores['r2']:.4f}, Best={best_r2:.4f}")
                        
                        # Early stopping if we hit 95%
                        if scores['r2'] > 0.95:
                            print(f"  🎉 EARLY STOP! Achieved {scores['r2']:.4f} R² (>95%)")
                            print(f"  ⚡ Optimal config: n_est={n_est}, lr={lr:.3f}, depth={depth}")
                            
                            self.best_configs[target] = best_config
                            return configs_tested, True, scores['r2']
                            
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        continue
        
        # Phase 2: Fine-tune around best config if not 95% yet
        if best_r2 < 0.95 and best_config:
            print(f"\n📊 PHASE 2: Fine-tuning around best config (R²={best_r2:.4f})")
            best_lr = best_config['learning_rate']
            best_depth = best_config['max_depth']
            
            # Test more n_estimators with best lr and depth
            n_estimators_fine = range(best_config['n_estimators'], min(401, best_config['n_estimators'] + 200), 25)
            
            for n_est in n_estimators_fine:
                configs_tested += 1
                
                try:
                    scores = self.evaluate_config_fast(n_est, best_lr, best_depth, X, y)
                    
                    result = {
                        'target': target,
                        'n_estimators': n_est,
                        'learning_rate': best_lr,
                        'max_depth': best_depth,
                        'r2_score': scores['r2'],
                        'rmse': scores['rmse'],
                        'mae': scores['mae'],
                        'training_time': scores['training_time']
                    }
                    
                    self.results.append(result)
                    
                    if scores['r2'] > best_r2:
                        best_r2 = scores['r2']
                        best_config = result.copy()
                        
                    print(f"  Fine-tune {configs_tested:3d}: n_est={n_est} -> R²={scores['r2']:.4f}, Best={best_r2:.4f}")
                    
                    if scores['r2'] > 0.95:
                        print(f"  🎉 EARLY STOP! Achieved {scores['r2']:.4f} R² (>95%)")
                        break
                        
                except Exception as e:
                    continue
        
        print(f"\n📊 SEARCH COMPLETE for {self.target_names[target]}:")
        print(f"  ✅ Configurations tested: {configs_tested}")
        print(f"  🏆 Best R² achieved: {best_r2:.4f} ({best_r2*100:.2f}%)")
        print(f"  ⚡ Best config: n_est={best_config['n_estimators']}, lr={best_config['learning_rate']:.3f}, depth={best_config['max_depth']}")
        
        if best_config:
            self.best_configs[target] = best_config
            
        return configs_tested, best_r2 > 0.95, best_r2
        
    def run_smart_search(self, df):
        """Run smart search for all targets"""
        print("🚀 STARTING SMART HYPERPARAMETER SEARCH")
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
        summary_stats = {}
        
        for target in self.target_columns:
            if target not in df.columns:
                continue
                
            X = df[available_features].fillna(0).values
            y = df[target].values
            
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            configs_tested, early_stopped, best_r2 = self.smart_search_for_target(target, X, y)
            summary_stats[target] = {
                'configs_tested': configs_tested,
                'early_stopped': early_stopped,
                'best_r2': best_r2
            }
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 SMART SEARCH COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Total configurations tested: {len(self.results)}")
        
        return summary_stats
        
    def create_results_summary(self, summary_stats):
        """Create comprehensive results summary"""
        print("\n📊 FINAL RESULTS SUMMARY")
        print("="*80)
        
        print(f"{'Rank':<4} {'Target Variable':<25} {'R² Score':<9} {'N Est':<6} {'LR':<6} {'Depth':<5} "
              f"{'RMSE':<8} {'Configs':<8} {'95%+ Hit':<8}")
        print("-" * 80)
        
        # Sort by R² score
        sorted_targets = sorted(self.best_configs.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        
        for rank, (target, config) in enumerate(sorted_targets, 1):
            early_stop = "YES" if summary_stats[target]['early_stopped'] else "NO"
            print(f"{rank:<4} {self.target_names[target]:<25} "
                  f"{config['r2_score']:<9.4f} {config['n_estimators']:<6} "
                  f"{config['learning_rate']:<6.3f} {config['max_depth']:<5} "
                  f"{config['rmse']:<8.3f} {summary_stats[target]['configs_tested']:<8} {early_stop:<8}")
        
        print("-" * 80)
        
        # Overall stats
        total_configs = sum(summary_stats[t]['configs_tested'] for t in summary_stats)
        early_stops = sum(1 for t in summary_stats if summary_stats[t]['early_stopped'])
        
        print(f"\n📈 EFFICIENCY SUMMARY:")
        print(f"  • Total configurations tested: {total_configs:,}")
        print(f"  • Targets achieving >95%: {early_stops}/{len(summary_stats)}")
        print(f"  • Average configs per target: {total_configs/len(summary_stats):.0f}")
        
        # Best performer
        if sorted_targets:
            best_target, best_config = sorted_targets[0]
            print(f"\n🏆 CHAMPION: {self.target_names[best_target]}")
            print(f"  • R² Score: {best_config['r2_score']:.4f} ({best_config['r2_score']*100:.2f}%)")
            print(f"  • Configuration: n_estimators={best_config['n_estimators']}, "
                  f"lr={best_config['learning_rate']:.3f}, depth={best_config['max_depth']}")
            print(f"  • Achieved >95%: {summary_stats[best_target]['early_stopped']}")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.output_dir, 'smart_search_results.csv'), index=False)
        
        # Save best configs
        with open(os.path.join(self.output_dir, 'best_configurations.json'), 'w') as f:
            json.dump(self.best_configs, f, indent=2)
            
        return results_df
        
    def run_complete_search(self):
        """Run the complete smart search"""
        print("🧠 SMART HYPERPARAMETER SEARCH WITH EARLY STOPPING")
        print("="*70)
        
        # Create data
        print("📊 Creating synthetic weather data...")
        df = self.create_synthetic_data()
        df = self.engineer_features(df)
        df = df.dropna(subset=self.target_columns)
        
        print(f"Dataset shape: {df.shape}")
        
        # Run search
        summary_stats = self.run_smart_search(df)
        
        # Create results
        results_df = self.create_results_summary(summary_stats)
        
        print(f"\n🎉 SEARCH COMPLETE! Results saved to: {self.output_dir}")
        return self.best_configs, results_df

def main():
    """Main function"""
    optimizer = SmartHyperparameterSearch()
    best_configs, results_df = optimizer.run_complete_search()
    
    print("\n" + "="*70)
    print("✅ SMART SEARCH COMPLETED!")
    print("="*70)
    
    return best_configs, results_df

if __name__ == "__main__":
    best_configs, results = main()
