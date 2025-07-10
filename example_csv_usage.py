#!/usr/bin/env python3
"""
Example script showing how to use CSV outputs from final_main_gbm.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def demonstrate_csv_outputs():
    """Demonstrate CSV output functionality"""
    print("="*60)
    print("GBM WEATHER PREDICTION - CSV OUTPUT DEMONSTRATION")
    print("="*60)
    
    # Look for the most recent CSV output directory
    csv_base_dir = os.path.join('gbm', 'csv_outputs')
    
    if not os.path.exists(csv_base_dir):
        print("❌ No CSV outputs found. Please run final_main_gbm.py first.")
        print("\nTo generate CSV outputs, run:")
        print("   python final_main_gbm.py")
        return
    
    # Find most recent output directory
    output_dirs = [d for d in os.listdir(csv_base_dir) if os.path.isdir(os.path.join(csv_base_dir, d))]
    if not output_dirs:
        print("❌ No output directories found.")
        return
    
    # Get most recent directory (assuming timestamp format)
    latest_dir = max(output_dirs)
    csv_dir = os.path.join(csv_base_dir, latest_dir)
    
    print(f"📁 Using CSV outputs from: {csv_dir}")
    print(f"📅 Generated on: {latest_dir}")
    print()
    
    # List available files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"📋 Available CSV files ({len(csv_files)}):")
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(csv_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"   {i}. {file} ({file_size} bytes)")
    print()
    
    # Demonstrate each CSV file
    demonstrate_feature_importance(csv_dir)
    demonstrate_prediction_samples(csv_dir)
    demonstrate_accuracy_metrics(csv_dir)
    demonstrate_metrics_summary(csv_dir)
    
    print("="*60)
    print("CSV DEMONSTRATION COMPLETED")
    print("="*60)

def demonstrate_feature_importance(csv_dir):
    """Demonstrate feature importance CSV"""
    file_path = os.path.join(csv_dir, 'feature_importance_all_targets.csv')
    
    if not os.path.exists(file_path):
        print("⚠️ Feature importance CSV not found")
        return
    
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Total features analyzed: {len(df)}")
        print(f"🔧 Engineered features: {len(df[df['Feature_Type'] == 'Engineered'])}")
        print(f"📋 Original features: {len(df[df['Feature_Type'] == 'Original'])}")
        print()
        
        print("🏆 TOP 10 MOST IMPORTANT FEATURES:")
        top_10 = df.head(10)
        for i, row in top_10.iterrows():
            feature_type_emoji = "🔧" if row['Feature_Type'] == 'Engineered' else "📋"
            print(f"   {row['Rank']:2d}. {feature_type_emoji} {row['Feature_Name']:20s} ({row['Average_Importance']:.4f})")
        
        print()
        print("📈 TARGET-SPECIFIC IMPORTANCE:")
        target_cols = [col for col in df.columns if col.endswith('_Importance') and col != 'Average_Importance']
        if target_cols:
            for target_col in target_cols:
                target_name = target_col.replace('_Importance', '')
                top_feature = df.loc[df[target_col].idxmax()]
                print(f"   {target_name}: {top_feature['Feature_Name']} ({top_feature[target_col]:.4f})")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading feature importance: {e}")

def demonstrate_prediction_samples(csv_dir):
    """Demonstrate prediction samples CSV"""
    # Look for prediction samples file
    pred_files = [f for f in os.listdir(csv_dir) if f.startswith('prediction_samples_')]
    
    if not pred_files:
        print("⚠️ Prediction samples CSV not found")
        return
    
    file_path = os.path.join(csv_dir, pred_files[0])
    
    print("🎯 PREDICTION SAMPLES ANALYSIS")
    print("-" * 40)
    
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Total samples: {len(df)}")
        print()
        
        # Show sample predictions
        print("📅 SAMPLE PREDICTIONS:")
        
        # Find target columns
        target_cols = []
        for col in df.columns:
            if col.endswith('_Actual'):
                target = col.replace('_Actual', '')
                if f'{target}_Predicted' in df.columns:
                    target_cols.append(target)
        
        print(f"🎯 Found predictions for: {', '.join(target_cols)}")
        print()
        
        # Show first 3 samples with details
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"📊 Sample {i+1}: {row['Date']} ({row['Day_of_Week']})")
            
            for target in target_cols[:3]:  # Limit to first 3 targets for readability
                actual_col = f'{target}_Actual'
                pred_col = f'{target}_Predicted'
                error_col = f'{target}_Absolute_Error'
                pct_error_col = f'{target}_Percentage_Error'
                
                if all(col in df.columns for col in [actual_col, pred_col, error_col]):
                    actual = row[actual_col]
                    predicted = row[pred_col]
                    error = row[error_col]
                    pct_error = row.get(pct_error_col, 0)
                    
                    print(f"   {target:10s}: Actual={actual:8.2f}, Predicted={predicted:8.2f}, Error={error:6.2f} ({pct_error:5.1f}%)")
            print()
        
        # Summary statistics
        print("📈 PREDICTION ACCURACY SUMMARY:")
        for target in target_cols:
            error_col = f'{target}_Absolute_Error'
            pct_error_col = f'{target}_Percentage_Error'
            
            if error_col in df.columns:
                avg_error = df[error_col].mean()
                max_error = df[error_col].max()
                avg_pct_error = df[pct_error_col].mean() if pct_error_col in df.columns else 0
                
                print(f"   {target:10s}: Avg Error={avg_error:6.2f}, Max Error={max_error:6.2f}, Avg %Error={avg_pct_error:5.1f}%")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading prediction samples: {e}")

def demonstrate_accuracy_metrics(csv_dir):
    """Demonstrate accuracy metrics CSV"""
    file_path = os.path.join(csv_dir, 'accuracy_metrics_all_models.csv')
    
    if not os.path.exists(file_path):
        print("⚠️ Accuracy metrics CSV not found")
        return
    
    print("📊 MODEL ACCURACY METRICS")
    print("-" * 40)
    
    try:
        df = pd.read_csv(file_path)
        print(f"🔢 Total model configurations: {len(df)}")
        print()
        
        # Group by target variable
        targets = df['Target_Variable'].unique()
        print(f"🎯 Target variables evaluated: {', '.join(targets)}")
        print()
        
        # Show metrics for each target
        print("📈 PERFORMANCE BY TARGET:")
        for target in targets:
            target_data = df[df['Target_Variable'] == target]
            
            if len(target_data) > 0:
                target_name = target_data.iloc[0]['Target_Name']
                best_r2 = target_data['R2_Score'].max()
                worst_r2 = target_data['R2_Score'].min()
                avg_r2 = target_data['R2_Score'].mean()
                best_rmse = target_data['RMSE'].min()
                worst_rmse = target_data['RMSE'].max()
                
                print(f"   {target_name:25s}:")
                print(f"      R² Score: Best={best_r2:6.3f}, Worst={worst_r2:6.3f}, Avg={avg_r2:6.3f}")
                print(f"      RMSE:     Best={best_rmse:6.3f}, Worst={worst_rmse:6.3f}")
                
                # Show model types
                model_types = target_data['Model_Type'].unique()
                print(f"      Models:   {', '.join(model_types)} ({len(target_data)} total)")
                print()
        
        # Show best performing models overall
        print("🏆 BEST PERFORMING MODELS:")
        best_models = df.loc[df.groupby('Target_Variable')['R2_Score'].idxmax()]
        for _, model in best_models.iterrows():
            horizon_label = model['Forecast_Label']
            print(f"   {model['Target_Name']:25s}: R²={model['R2_Score']:6.3f}, RMSE={model['RMSE']:6.3f} ({horizon_label})")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading accuracy metrics: {e}")

def demonstrate_metrics_summary(csv_dir):
    """Demonstrate metrics summary CSV"""
    file_path = os.path.join(csv_dir, 'accuracy_metrics_summary.csv')
    
    if not os.path.exists(file_path):
        print("⚠️ Metrics summary CSV not found")
        return
    
    print("📋 METRICS SUMMARY")
    print("-" * 40)
    
    try:
        df = pd.read_csv(file_path)
        
        # Performance ranking
        df_sorted = df.sort_values('Best_R2_Score', ascending=False)
        
        print("🏆 TARGET VARIABLE RANKING (by Best R² Score):")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            performance_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            
            print(f"   {i}. {performance_emoji} {row['Target_Name']:25s}")
            print(f"      Best R²: {row['Best_R2_Score']:6.3f}, Avg R²: {row['Average_R2_Score']:6.3f}")
            print(f"      Best RMSE: {row['Best_RMSE']:6.3f}, Avg RMSE: {row['Average_RMSE']:6.3f}")
            print(f"      Models: {row['Number_of_Models']}")
            print()
        
        # Overall statistics
        print("🌟 OVERALL STATISTICS:")
        print(f"   Average R² across all targets: {df['Average_R2_Score'].mean():.3f}")
        print(f"   Best overall R² score: {df['Best_R2_Score'].max():.3f}")
        print(f"   Total models trained: {df['Number_of_Models'].sum()}")
        
        # Performance categories
        excellent = len(df[df['Best_R2_Score'] >= 0.7])
        good = len(df[(df['Best_R2_Score'] >= 0.4) & (df['Best_R2_Score'] < 0.7)])
        poor = len(df[df['Best_R2_Score'] < 0.4])
        
        print(f"   Performance distribution:")
        print(f"      Excellent (R² ≥ 0.7): {excellent} targets")
        print(f"      Good (0.4 ≤ R² < 0.7): {good} targets")
        print(f"      Needs improvement (R² < 0.4): {poor} targets")
        
        print()
        
    except Exception as e:
        print(f"❌ Error reading metrics summary: {e}")

def analyze_csv_for_research():
    """Provide research-oriented analysis of CSV outputs"""
    print("🔬 RESEARCH INSIGHTS FROM CSV DATA")
    print("-" * 50)
    
    csv_base_dir = os.path.join('gbm', 'csv_outputs')
    if not os.path.exists(csv_base_dir):
        return
    
    output_dirs = [d for d in os.listdir(csv_base_dir) if os.path.isdir(os.path.join(csv_base_dir, d))]
    if not output_dirs:
        return
    
    latest_dir = max(output_dirs)
    csv_dir = os.path.join(csv_base_dir, latest_dir)
    
    # Research insights
    insights = []
    
    # Feature engineering effectiveness
    feature_file = os.path.join(csv_dir, 'feature_importance_all_targets.csv')
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        engineered_in_top10 = len(df.head(10)[df.head(10)['Feature_Type'] == 'Engineered'])
        total_engineered = len(df[df['Feature_Type'] == 'Engineered'])
        total_original = len(df[df['Feature_Type'] == 'Original'])
        
        insights.append(f"🔧 Feature Engineering Impact: {engineered_in_top10}/10 top features are engineered")
        insights.append(f"📊 Feature Distribution: {total_engineered} engineered vs {total_original} original features")
    
    # Model performance insights
    metrics_file = os.path.join(csv_dir, 'accuracy_metrics_summary.csv')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        best_target = df.loc[df['Best_R2_Score'].idxmax()]
        worst_target = df.loc[df['Best_R2_Score'].idxmin()]
        
        insights.append(f"🏆 Best performing target: {best_target['Target_Name']} (R² = {best_target['Best_R2_Score']:.3f})")
        insights.append(f"📉 Most challenging target: {worst_target['Target_Name']} (R² = {worst_target['Best_R2_Score']:.3f})")
    
    # Print insights
    for insight in insights:
        print(f"   {insight}")
    
    print()
    
    # Recommendations
    print("💡 RESEARCH RECOMMENDATIONS:")
    print("   1. Include feature importance analysis in methodology section")
    print("   2. Highlight effectiveness of engineered features")
    print("   3. Discuss performance variations across target variables")
    print("   4. Use prediction samples for case study analysis")
    print("   5. Compare single-day vs multi-day forecasting performance")
    print()

if __name__ == "__main__":
    demonstrate_csv_outputs()
    analyze_csv_for_research() 