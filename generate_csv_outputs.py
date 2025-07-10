#!/usr/bin/env python3
"""
Standalone script to generate CSV output tables from trained GBM models
This script can be run independently to generate CSV outputs from existing models and data.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
import warnings

# Import our modular components
from gbm_config import *
from gbm_data_preprocessing import DataPreprocessor
from gbm_model_training import GBMModelTrainer
from gbm_output_generator import GBMOutputGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def load_trained_models():
    """Load trained models from the models directory"""
    try:
        print("Loading trained models...")
        
        target_models = {}
        
        for target in TARGET_COLUMNS:
            target_dir = os.path.join(MODELS_DIR, f'target_{target}')
            
            if not os.path.exists(target_dir):
                print(f"Warning: Model directory for {target} not found: {target_dir}")
                continue
            
            model_path = os.path.join(target_dir, f'{target}_model.pkl')
            scaler_path = os.path.join(target_dir, f'{target}_scaler.pkl')
            feature_scaler_path = os.path.join(target_dir, f'{target}_feature_scaler.pkl')
            metrics_path = os.path.join(target_dir, f'{target}_metrics.pkl')
            
            if not os.path.exists(model_path):
                print(f"Warning: Model file for {target} not found: {model_path}")
                continue
            
            # Load model
            model = joblib.load(model_path)
            
            # Load scaler (if exists)
            scaler = None
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            
            # Load metrics (if exists)
            metrics = {}
            if os.path.exists(metrics_path):
                metrics = joblib.load(metrics_path)
            
            target_models[target] = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics
            }
            
            print(f"  Loaded {target} model successfully")
        
        # Load feature scaler
        feature_scaler_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
        if os.path.exists(feature_scaler_path):
            feature_columns = joblib.load(feature_scaler_path)
            print(f"  Loaded feature columns: {len(feature_columns)} features")
        
        # Try to load the main feature scaler (from any target directory)
        main_scaler = None
        for target in TARGET_COLUMNS:
            scaler_path = os.path.join(MODELS_DIR, f'target_{target}', f'{target}_feature_scaler.pkl')
            if os.path.exists(scaler_path):
                main_scaler = joblib.load(scaler_path)
                print(f"  Loaded feature scaler from {target} directory")
                break
        
        return target_models, main_scaler
        
    except Exception as e:
        print(f"Error loading trained models: {str(e)}")
        raise

def generate_csv_outputs_from_existing(data_path='makassar.csv', n_prediction_samples=10):
    """Generate CSV outputs from existing trained models"""
    try:
        print("=" * 60)
        print("GBM CSV OUTPUT GENERATOR")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n1. Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)
        print(f"   Preprocessing completed: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        
        # Step 3: Load trained models
        print("\n3. Loading trained models...")
        target_models, feature_scaler = load_trained_models()
        
        if not target_models:
            print("Error: No trained models found. Please train models first using gbm_main.py")
            return None
        
        if feature_scaler is None:
            print("Warning: Feature scaler not found. Creating new scaler...")
            available_features = [col for col in FEATURE_COLUMNS if col in processed_df.columns]
            X = processed_df[available_features]
            feature_scaler = StandardScaler()
            feature_scaler.fit(X)
        
        # Step 4: Generate CSV outputs
        print("\n4. Generating CSV output tables...")
        output_generator = GBMOutputGenerator()
        
        csv_outputs = output_generator.generate_all_tables(
            processed_df, 
            target_models, 
            feature_scaler, 
            multi_day_results=None,  # Set to None if no multi-day results available
            n_prediction_samples=n_prediction_samples
        )
        
        print("\n" + "=" * 60)
        print("CSV OUTPUT GENERATION COMPLETED")
        print("=" * 60)
        print(f"CSV outputs saved to: {output_generator.output_dir}")
        
        # Display file list
        print("\nGenerated files:")
        output_files = os.listdir(output_generator.output_dir)
        for i, file in enumerate(sorted(output_files), 1):
            print(f"  {i}. {file}")
        
        return csv_outputs
        
    except Exception as e:
        print(f"Error generating CSV outputs: {str(e)}")
        raise

def generate_csv_with_retrain(data_path='makassar.csv', n_prediction_samples=10):
    """Generate CSV outputs by retraining models (if no existing models)"""
    try:
        print("No existing models found. Training new models...")
        
        # Import and run the main predictor
        from gbm_main import GBMWeatherPredictor
        
        predictor = GBMWeatherPredictor()
        results = predictor.run_complete_pipeline(data_path)
        
        print("\nModel training and CSV generation completed!")
        return results['csv_outputs']
        
    except Exception as e:
        print(f"Error in retraining and CSV generation: {str(e)}")
        raise

def main():
    """Main function to generate CSV outputs"""
    try:
        # Check command line arguments
        data_path = 'makassar.csv'
        n_samples = 10
        
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
        if len(sys.argv) > 2:
            n_samples = int(sys.argv[2])
        
        print(f"Data file: {data_path}")
        print(f"Number of prediction samples: {n_samples}")
        
        # Check if trained models exist
        models_exist = any(
            os.path.exists(os.path.join(MODELS_DIR, f'target_{target}'))
            for target in TARGET_COLUMNS
        )
        
        if models_exist:
            print("\nFound existing trained models. Generating CSV outputs...")
            results = generate_csv_outputs_from_existing(data_path, n_samples)
        else:
            print("\nNo existing models found. Training new models...")
            results = generate_csv_with_retrain(data_path, n_samples)
        
        if results:
            print("\n" + "="*50)
            print("SUCCESS: CSV outputs generated successfully!")
            print("="*50)
            
            # Print brief summary
            print("\nGenerated Tables:")
            print("1. Feature Importance for All Targets")
            print("2. Top 10 Feature Importance Summary")
            print(f"3. Prediction Results for First {n_samples} Samples")
            print("4. Model Accuracy Metrics")
            print("5. Model Performance Summary")
            print("6. Combined Summary Report")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    print("GBM Weather Prediction - CSV Output Generator")
    print("Usage: python generate_csv_outputs.py [data_file.csv] [num_prediction_samples]")
    print("Example: python generate_csv_outputs.py makassar.csv 10")
    print("")
    
    main() 