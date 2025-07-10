#!/usr/bin/env python3
"""
Test script for CSV output functionality
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings

# Import the main class
from final_main_gbm import GBMWeatherPredictor

# Suppress warnings
warnings.filterwarnings('ignore')

def test_csv_outputs():
    """Test CSV output generation with a smaller dataset"""
    try:
        print("=== Testing CSV Output Generation ===")
        
        # Load a small sample of the dataset
        print("Loading dataset...")
        df = pd.read_csv('makassar.csv')
        
        # Take only first 100 rows for quick testing
        df_sample = df.head(100).copy()
        print(f"Using sample dataset: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns")
        
        # Initialize predictor
        predictor = GBMWeatherPredictor()
        
        # Quick preprocessing
        print("Preprocessing data...")
        processed_df = predictor.preprocess_data(df_sample)
        print(f"Processed shape: {processed_df.shape}")
        
        # Quick training (single target for testing)
        print("Training simplified model...")
        
        # Prepare feature matrix
        X = processed_df[predictor.feature_columns]
        
        # Just train one model for testing
        target = 'RR'  # Rainfall
        y = processed_df[target]
        
        # Simple split
        train_size = int(0.8 * len(processed_df))
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
        
        # Scale features
        X_train_scaled = predictor.scaler.fit_transform(X_train)
        X_test_scaled = predictor.scaler.transform(X_test)
        
        # Simple model
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred = np.maximum(y_pred, 0)  # Rainfall constraint
        
        # Store in predictor format
        predictor.multi_target_models[target] = {
            'model': model,
            'scaler': None,
            'metrics': {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        }
        
        print(f"Model trained. R²: {predictor.multi_target_models[target]['metrics']['R2']:.4f}")
        
        # Test CSV generation
        print("\nTesting CSV output generation...")
        
        # Create test directory
        csv_output_dir = os.path.join('gbm', 'csv_outputs', 'test_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(csv_output_dir, exist_ok=True)
        
        # Test 1: Feature importance CSV
        print("1. Testing feature importance CSV...")
        predictor.generate_feature_importance_csv(csv_output_dir)
        
        # Test 2: Prediction samples CSV (5 samples)
        print("2. Testing prediction samples CSV...")
        predictor.generate_prediction_samples_csv(processed_df, csv_output_dir, n_samples=5)
        
        # Test 3: Accuracy metrics CSV
        print("3. Testing accuracy metrics CSV...")
        predictor.generate_accuracy_metrics_csv(output_dir=csv_output_dir)
        
        # List generated files
        print(f"\nGenerated files in {csv_output_dir}:")
        for file in os.listdir(csv_output_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  - {file} ({file_size} bytes)")
                
                # Show first few rows
                try:
                    df_check = pd.read_csv(file_path)
                    print(f"    Shape: {df_check.shape}")
                    print(f"    Columns: {list(df_check.columns)}")
                except Exception as e:
                    print(f"    Error reading file: {e}")
        
        print("\n=== CSV Output Test Completed Successfully ===")
        return csv_output_dir
        
    except Exception as e:
        print(f"Error in CSV output test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_csv_outputs() 