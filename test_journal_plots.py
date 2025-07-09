#!/usr/bin/env python3
"""
Test script to verify journal plot generation works correctly
"""

import sys
import os
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_journal_plotter():
    """Test the journal plotter functionality"""
    try:
        print("🧪 Testing Journal Plotter...")
        
        # Import the journal plotter
        from gbm_weather_predictor.visualization.journal_plotter import JournalPlotter
        
        # Create journal plotter
        plotter = JournalPlotter("test_journal_plots")
        print("✅ JournalPlotter instantiated successfully")
        
        # Load sample data
        print("📊 Loading sample data...")
        df = pd.read_csv('makassar.csv')
        
        # Basic preprocessing
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        df['Month'] = df['Tanggal'].dt.month
        
        # Test creating a simple plot
        print("🎨 Creating test plot...")
        # Ensure Month column exists for the plot
        if 'Month' not in df.columns:
            df['Month'] = df['Tanggal'].dt.month
        plotter.create_target_distribution_plot(df)
        
        print("✅ Journal plotter test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Journal plotter test failed: {str(e)}")
        return False

def test_full_pipeline():
    """Test the full pipeline with journal plots"""
    try:
        print("\n🚀 Testing Full Pipeline with Journal Plots...")
        
        from gbm_weather_predictor import GBMWeatherPredictor
        from gbm_weather_predictor.data import DataPreprocessor, FeatureEngineer
        
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        df = pd.read_csv('makassar.csv')
        
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)
        
        # Engineer features
        print("🔧 Engineering features...")
        feature_engineer = FeatureEngineer()
        processed_df = feature_engineer.engineer_features(processed_df)
        
        # Train models (with journal plots)
        print("🤖 Training models and generating journal plots...")
        predictor = GBMWeatherPredictor()
        metrics = predictor.train_and_evaluate(processed_df)
        
        print("✅ Full pipeline with journal plots completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Journal Plot Generation")
    print("=" * 50)
    
    tests = [
        test_journal_plotter,
        test_full_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Journal plots are working correctly.")
        print("📁 Check the 'journal_plots/' directory for publication-ready figures.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 