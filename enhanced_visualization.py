#!/usr/bin/env python3
"""
Enhanced visualization for hyperparameter optimization results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Load results
results_dir = "hyperparameter_optimization_20250620_161557"
results_df = pd.read_csv(f"{results_dir}/all_results.csv")
with open(f"{results_dir}/best_configurations.json", 'r') as f:
    best_configs = json.load(f)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a comprehensive dashboard
fig = plt.figure(figsize=(20, 16))

# 1. Best R² Scores by Target (Top Left)
ax1 = plt.subplot(3, 4, 1)
target_names = {
    'RR': 'Rainfall',
    'ss': 'Sunshine',
    'Tavg': 'Temperature',
    'ddd_car': 'Wind Dir',
    'ff_avg': 'Wind Speed'
}

best_r2_data = []
for target, config in best_configs.items():
    best_r2_data.append({
        'target': target_names[target],
        'r2_score': config['r2_score']
    })

best_r2_df = pd.DataFrame(best_r2_data)
colors = ['green' if x > 0.8 else 'orange' if x > 0.2 else 'red' for x in best_r2_df['r2_score']]
bars = ax1.bar(best_r2_df['target'], best_r2_df['r2_score'], color=colors, alpha=0.7)
ax1.set_title('Best R² Score by Target Variable', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim(-0.1, 1.0)
for i, (bar, score) in enumerate(zip(bars, best_r2_df['r2_score'])):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=45)

# 2. Learning Rate Impact (Top Center-Left)
ax2 = plt.subplot(3, 4, 2)
for target in results_df['target'].unique():
    target_data = results_df[results_df['target'] == target]
    ax2.scatter(target_data['learning_rate'], target_data['r2_score'], 
               label=target_names.get(target, target), alpha=0.6, s=30)
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('R² Score')
ax2.set_title('Learning Rate Impact on R² Score', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. N Estimators Impact (Top Center-Right)
ax3 = plt.subplot(3, 4, 3)
for target in results_df['target'].unique():
    target_data = results_df[results_df['target'] == target]
    ax3.scatter(target_data['n_estimators'], target_data['r2_score'], 
               label=target_names.get(target, target), alpha=0.6, s=30)
ax3.set_xlabel('Number of Estimators')
ax3.set_ylabel('R² Score')
ax3.set_title('N Estimators Impact on R² Score', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Max Depth Impact (Top Right)
ax4 = plt.subplot(3, 4, 4)
for target in results_df['target'].unique():
    target_data = results_df[results_df['target'] == target]
    ax4.scatter(target_data['max_depth'], target_data['r2_score'], 
               label=target_names.get(target, target), alpha=0.6, s=30)
ax4.set_xlabel('Max Depth')
ax4.set_ylabel('R² Score')
ax4.set_title('Max Depth Impact on R² Score', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Best Configuration Table (Middle Left - spans 2 columns)
ax5 = plt.subplot(3, 4, (5, 6))
ax5.axis('off')

# Create table data
table_data = []
for target, config in best_configs.items():
    table_data.append([
        target_names[target],
        f"{config['n_estimators']}",
        f"{config['learning_rate']:.3f}",
        f"{config['max_depth']}",
        f"{config['r2_score']:.4f}",
        f"{config['rmse']:.3f}"
    ])

table = ax5.table(cellText=table_data,
                 colLabels=['Target', 'N Est', 'LR', 'Depth', 'R² Score', 'RMSE'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0.3, 1, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color cells based on performance
for i in range(len(table_data)):
    r2_score = float(table_data[i][4])
    if r2_score > 0.8:
        color = 'lightgreen'
    elif r2_score > 0.2:
        color = 'lightyellow'
    else:
        color = 'lightcoral'
    
    for j in range(len(table_data[i])):
        table[(i+1, j)].set_facecolor(color)

ax5.set_title('Best Configurations Summary', fontsize=14, fontweight='bold', y=0.8)

# 6. Training Time vs Accuracy (Middle Center-Right)
ax6 = plt.subplot(3, 4, 7)
for target in results_df['target'].unique():
    target_data = results_df[results_df['target'] == target]
    ax6.scatter(target_data['training_time'], target_data['r2_score'], 
               label=target_names.get(target, target), alpha=0.6, s=30)
ax6.set_xlabel('Training Time (seconds)')
ax6.set_ylabel('R² Score')
ax6.set_title('Training Time vs Accuracy Trade-off', fontsize=12, fontweight='bold')
ax6.set_xscale('log')
ax6.grid(True, alpha=0.3)

# 7. RMSE Distribution (Middle Right)
ax7 = plt.subplot(3, 4, 8)
rmse_data = []
labels = []
for target in ['RR', 'ss', 'Tavg']:  # Skip wind variables for better scale
    target_data = results_df[results_df['target'] == target]
    rmse_data.append(target_data['rmse'].values)
    labels.append(target_names[target])

bp = ax7.boxplot(rmse_data, labels=labels, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax7.set_ylabel('RMSE')
ax7.set_title('RMSE Distribution by Target', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# 8. Performance Heatmap for Temperature (Bottom Left)
ax8 = plt.subplot(3, 4, 9)
temp_data = results_df[results_df['target'] == 'Tavg']
pivot_temp = temp_data.pivot_table(values='r2_score', index='max_depth', 
                                  columns='learning_rate', aggfunc='mean')
sns.heatmap(pivot_temp, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax8)
ax8.set_title('Temperature: R² by Depth vs LR', fontsize=12, fontweight='bold')

# 9. Performance Heatmap for Rainfall (Bottom Center-Left)
ax9 = plt.subplot(3, 4, 10)
rain_data = results_df[results_df['target'] == 'RR']
pivot_rain = rain_data.pivot_table(values='r2_score', index='max_depth', 
                                  columns='learning_rate', aggfunc='mean')
sns.heatmap(pivot_rain, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax9)
ax9.set_title('Rainfall: R² by Depth vs LR', fontsize=12, fontweight='bold')

# 10. Optimization Summary Stats (Bottom Center-Right)
ax10 = plt.subplot(3, 4, 11)
ax10.axis('off')

# Summary statistics
total_configs = len(results_df)
best_overall = results_df['r2_score'].max()
worst_overall = results_df['r2_score'].min()
avg_training_time = results_df['training_time'].mean()

summary_text = f"""
OPTIMIZATION SUMMARY
═══════════════════════
Total Configurations: {total_configs}
Best R² Score: {best_overall:.4f}
Worst R² Score: {worst_overall:.4f}
Avg Training Time: {avg_training_time:.3f}s

PERFORMANCE LEVELS
═══════════════════════
Excellent (R² > 0.8): {len(results_df[results_df['r2_score'] > 0.8])}
Good (0.5 < R² ≤ 0.8): {len(results_df[(results_df['r2_score'] > 0.5) & (results_df['r2_score'] <= 0.8)])}
Poor (R² ≤ 0.5): {len(results_df[results_df['r2_score'] <= 0.5])}

TOP PARAMETERS
═══════════════════════
Best Learning Rate: 0.1-0.15
Best Max Depth: 3
Best N Estimators: 300-400
"""

ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

# 11. Configuration Distribution (Bottom Right)
ax11 = plt.subplot(3, 4, 12)
# Create a simplified parameter combination frequency plot
param_combos = results_df.groupby(['learning_rate', 'max_depth'])['r2_score'].mean().reset_index()
scatter = ax11.scatter(param_combos['learning_rate'], param_combos['max_depth'], 
                      c=param_combos['r2_score'], s=100, cmap='RdYlGn', alpha=0.7)
ax11.set_xlabel('Learning Rate')
ax11.set_ylabel('Max Depth')
ax11.set_title('Avg R² by LR-Depth Combination', fontsize=12, fontweight='bold')
ax11.set_xscale('log')
plt.colorbar(scatter, ax=ax11, label='Avg R² Score')

# Overall title
fig.suptitle('Hyperparameter Optimization Dashboard - 100 Configurations Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('enhanced_optimization_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("Enhanced dashboard saved as 'enhanced_optimization_dashboard.png'")
