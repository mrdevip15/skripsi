#!/usr/bin/env python3
"""
Test script to verify the organized GBM Weather Predictor structure works correctly
"""

import sys
import os
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly"""
    try:
        print("Testing imports...")
        
        # Test config imports
        from gbm_weather_predictor.config.settings import TARGET_COLUMNS, TARGET_NAMES, MODEL_CONFIGS
        print("✅ Config imports successful")
        
        # Test data imports
        from gbm_weather_predictor.data.preprocessing import DataPreprocessor
        from gbm_weather_predictor.data.feature_engineering import FeatureEngineer
        print("✅ Data imports successful")
        
        # Test models imports
        from gbm_weather_predictor.models.gbm_predictor import GBMWeatherPredictor
        print("✅ Models imports successful")
        
        # Test visualization imports
        from gbm_weather_predictor.visualization.plotting import WeatherPlotter
        print("✅ Visualization imports successful")
        
        # Test utils imports
        from gbm_weather_predictor.utils.helpers import create_timestamp, sanitize_filename
        print("✅ Utils imports successful")
        
        # Test main package import
        from gbm_weather_predictor import GBMWeatherPredictor as GBM
        print("✅ Main package import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

def test_components():
    """Test that components can be instantiated"""
    try:
        print("\nTesting component instantiation...")
        
        # Import components
        from gbm_weather_predictor.data.preprocessing import DataPreprocessor
        from gbm_weather_predictor.data.feature_engineering import FeatureEngineer
        from gbm_weather_predictor.models.gbm_predictor import GBMWeatherPredictor
        from gbm_weather_predictor.visualization.plotting import WeatherPlotter
        
        # Test preprocessor
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor instantiated")
        
        # Test feature engineer
        feature_engineer = FeatureEngineer()
        print("✅ FeatureEngineer instantiated")
        
        # Test predictor
        predictor = GBMWeatherPredictor()
        print("✅ GBMWeatherPredictor instantiated")
        
        # Test plotter
        plotter = WeatherPlotter("test_output")
        print("✅ WeatherPlotter instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

def test_configuration():
    """Test that configuration is properly loaded"""
    try:
        print("\nTesting configuration...")
        
        from gbm_weather_predictor.config.settings import TARGET_COLUMNS, TARGET_NAMES, MODEL_CONFIGS
        
        # Check target columns
        expected_targets = ['RR', 'ss', 'Tavg', 'ddd_car', 'ff_avg']
        assert TARGET_COLUMNS == expected_targets, f"Expected {expected_targets}, got {TARGET_COLUMNS}"
        print("✅ Target columns configuration correct")
        
        # Check target names
        assert 'RR' in TARGET_NAMES, "RR not found in TARGET_NAMES"
        assert 'ss' in TARGET_NAMES, "ss not found in TARGET_NAMES"
        print("✅ Target names configuration correct")
        
        # Check model configs
        assert 'RR' in MODEL_CONFIGS, "RR model config not found"
        assert 'n_estimators' in MODEL_CONFIGS['RR'], "RR model config missing n_estimators"
        print("✅ Model configurations correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Organized GBM Weather Predictor Structure")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_components,
        test_configuration
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
        print("🎉 All tests passed! The organized structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 