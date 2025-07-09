#!/usr/bin/env python3
"""
Runner script for the organized GBM Weather Predictor with Journal Plots
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gbm_weather_predictor.main import main

if __name__ == "__main__":
    print("🚀 Starting GBM Weather Predictor with Journal Plot Generation...")
    print("📊 This will create publication-ready plots automatically!")
    main()
    print("\n🎨 Journal plots have been generated in the 'journal_plots/' directory!")
    print("📈 Ready for publication!") 