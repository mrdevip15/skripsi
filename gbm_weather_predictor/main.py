"""
Main entry point for GBM Weather Predictor
"""

import pandas as pd
import matplotlib.pyplot as plt
from .models.gbm_predictor import GBMWeatherPredictor
from .data.preprocessing import DataPreprocessor
from .data.feature_engineering import FeatureEngineer

def main():
    """Main function to run the complete weather prediction pipeline"""
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv('makassar.csv')
        
        # Initialize predictor
        predictor = GBMWeatherPredictor()
        
        # Preprocess data
        print("Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)
        
        # Engineer features
        print("Engineering features...")
        feature_engineer = FeatureEngineer()
        processed_df = feature_engineer.engineer_features(processed_df)
        
        # Train and evaluate - this will also generate all plots
        print("Training and evaluating model...")
        predictor.train_and_evaluate(processed_df)
        
        # Save models
        print("Saving models...")
        predictor.save_model()
        
        print("Process completed successfully.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        # Clean up matplotlib resources
        plt.close('all')

if __name__ == "__main__":
    main() 