#!/usr/bin/env python3
"""
Test script untuk menjalankan sistem prediksi cuaca dengan TimeSeriesSplit
dan menghasilkan visualisasi split data untuk semua target parameter.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import the main class
from final_main_gbm import GBMWeatherPredictor

def test_timeseries_split():
    """Test the TimeSeriesSplit implementation and data splits visualization"""
    
    try:
        print("=== Testing TimeSeriesSplit Implementation ===")
        
        # Check if dataset exists
        if not os.path.exists('makassar.csv'):
            print("Error: makassar.csv not found. Please ensure the dataset is available.")
            return
        
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv('makassar.csv')
        print(f"Dataset loaded: {len(df)} samples")
        
        # Initialize predictor
        print("Initializing GBM Weather Predictor...")
        predictor = GBMWeatherPredictor()
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Generate data splits visualization
        print("Generating data splits visualization...")
        predictor.plot_data_splits(processed_df)
        
        print("\n=== Test completed successfully ===")
        print("Check the plots directory for the data splits visualization:")
        print(f"Directory: {predictor.run_dir}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timeseries_split() 