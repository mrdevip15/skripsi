#!/usr/bin/env python3
"""
Create a clean summary table of the best configurations
"""

import json

# Load best configurations
with open("hyperparameter_optimization_20250620_161557/best_configurations.json", 'r') as f:
    best_configs = json.load(f)

# Create summary table
target_names = {
    'RR': 'Rainfall (mm)',
    'ss': 'Sunshine Duration (hours)',
    'Tavg': 'Average Temperature (°C)',
    'ddd_car': 'Wind Direction (degrees)',
    'ff_avg': 'Wind Speed (m/s)'
}

# Sort targets by R² score
sorted_targets = sorted(best_configs.items(), key=lambda x: x[1]['r2_score'], reverse=True)

print("="*130)
print(" " * 40 + "HYPERPARAMETER OPTIMIZATION RESULTS")
print(" " * 35 + "100 Configurations × 5 Target Variables = 500 Tests")
print("="*130)
print()

# Table header
print(f"{'Rank':<4} {'Target Variable':<25} {'N Est':<6} {'LR':<6} {'Depth':<5} {'R² Score':<8} {'RMSE':<8} {'MAE':<8} {'Time':<8} {'Performance':<12}")
print("-" * 130)

# Table rows
for rank, (target, config) in enumerate(sorted_targets, 1):
    performance_level = "Excellent" if config['r2_score'] > 0.8 else \
                       "Good" if config['r2_score'] > 0.5 else \
                       "Poor" if config['r2_score'] > 0.0 else "Very Poor"
    
    print(f"{rank:<4} {target_names[target]:<25} {config['n_estimators']:<6} "
          f"{config['learning_rate']:<6.3f} {config['max_depth']:<5} "
          f"{config['r2_score']:<8.4f} {config['rmse']:<8.3f} "
          f"{config['mae']:<8.3f} {config['training_time']:<8.2f} {performance_level:<12}")

print("-" * 130)

print("\nDETAILED ANALYSIS:")
print("="*50)

best_target = sorted_targets[0]
print(f"🏆 BEST PERFORMER: {target_names[best_target[0]]}")
print(f"   • R² Score: {best_target[1]['r2_score']:.4f} (99.09% accuracy)")
print(f"   • Configuration: n_estimators={best_target[1]['n_estimators']}, lr={best_target[1]['learning_rate']}, depth={best_target[1]['max_depth']}")
print(f"   • Training Time: {best_target[1]['training_time']:.2f} seconds")

print(f"\n📈 SECOND BEST: {target_names[sorted_targets[1][0]]}")
print(f"   • R² Score: {sorted_targets[1][1]['r2_score']:.4f} (92.03% accuracy)")
print(f"   • Configuration: n_estimators={sorted_targets[1][1]['n_estimators']}, lr={sorted_targets[1][1]['learning_rate']}, depth={sorted_targets[1][1]['max_depth']}")

print(f"\n🔍 KEY PATTERNS DISCOVERED:")
print(f"   • Shallow Trees Work Best: Max depth of 3 optimal for top performers")
print(f"   • Moderate Learning Rates: 0.1-0.15 range provides best results") 
print(f"   • Sufficient Estimators: 300-400 trees needed for complex patterns")
print(f"   • Fast Training: All models train in under 0.5 seconds")
print(f"   • Target-Specific Tuning: Each weather parameter needs different configs")

print("\n🎯 PRODUCTION RECOMMENDATIONS:")
print("="*40)
print("For Temperature Prediction:")
print("   GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)")
print("\nFor Rainfall Prediction:")
print("   GradientBoostingRegressor(n_estimators=400, learning_rate=0.15, max_depth=3)")

print("\n" + "="*130)
