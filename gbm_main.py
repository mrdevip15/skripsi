#!/usr/bin/env python3
"""
Main execution file for GBM Weather Prediction System
This file orchestrates the entire weather prediction pipeline using modular components.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings

# Import our modular components
from gbm_config import *
from gbm_data_preprocessing import DataPreprocessor
from gbm_model_training import GBMModelTrainer
from gbm_visualization import GBMVisualizer
from gbm_output_generator import GBMOutputGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

class GBMWeatherPredictor:
    """Main orchestrator class for the GBM weather prediction system"""
    
    def __init__(self):
        """Initialize the GBM weather predictor with all components"""
        self.preprocessor = DataPreprocessor()
        self.trainer = GBMModelTrainer()
        self.visualizer = GBMVisualizer()
        self.output_generator = GBMOutputGenerator()
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(PLOTS_DIR, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"GBM Weather Predictor initialized")
        print(f"Run timestamp: {self.timestamp}")
        print(f"Output directory: {self.run_dir}")
        print(f"CSV outputs directory: {self.output_generator.output_dir}")
    
    def run_complete_pipeline(self, data_path='makassar.csv'):
        """Run the complete weather prediction pipeline"""
        try:
            print("=" * 60)
            print("GBM WEATHER PREDICTION SYSTEM")
            print("=" * 60)
            
            # Step 1: Load dataset
            print("\n1. Loading dataset...")
            df = pd.read_csv(data_path)
            print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Step 2: Preprocess data
            print("\n2. Preprocessing data...")
            processed_df = self.preprocessor.preprocess_data(df)
            print(f"   Preprocessing completed: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
            
            # Step 3: Generate initial visualizations
            print("\n3. Generating initial visualizations...")
            self.visualizer.plot_target_distributions(processed_df)
            self.visualizer.plot_feature_correlations(processed_df)
            self.visualizer.plot_feature_distributions(processed_df)
            self.visualizer.plot_seasonal_patterns(processed_df)
            self.visualizer.plot_data_splits(processed_df)
            
            # Step 4: Train and evaluate models
            print("\n4. Training and evaluating models...")
            metrics = self.trainer.train_and_evaluate(processed_df)
            
            # Step 5: Multi-day forecasting
            print("\n5. Multi-day forecasting...")
            multi_day_data = self.preprocessor.prepare_multi_day_dataset(processed_df)
            multi_day_results = self.trainer.train_multi_day_models(multi_day_data)
            
            # Step 6: Save models
            print("\n6. Saving models...")
            self.trainer.save_multi_target_models()
            
            # Step 7: Generate CSV output tables
            print("\n7. Generating CSV output tables...")
            csv_outputs = self.output_generator.generate_all_tables(
                processed_df, 
                self.trainer.multi_target_models, 
                self.trainer.scaler, 
                multi_day_results, 
                n_prediction_samples=10
            )
            
            # Step 8: Generate final summary
            print("\n8. Generating final summary...")
            self._generate_final_summary(metrics, multi_day_results)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Results saved to: {self.run_dir}")
            print(f"Models saved to: {MODELS_DIR}")
            print(f"CSV outputs saved to: {self.output_generator.output_dir}")
            
            return {
                'metrics': metrics,
                'multi_day_results': multi_day_results,
                'run_dir': self.run_dir,
                'csv_outputs': csv_outputs
            }
            
        except Exception as e:
            print(f"\nError in pipeline execution: {str(e)}")
            raise
    
    def _generate_final_summary(self, metrics, multi_day_results):
        """Generate a final summary of all results"""
        try:
            summary_file = os.path.join(self.run_dir, 'final_summary.txt')
            
            with open(summary_file, 'w') as f:
                f.write("GBM WEATHER PREDICTION SYSTEM - FINAL SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Run Timestamp: {self.timestamp}\n")
                f.write(f"Output Directory: {self.run_dir}\n\n")
                
                # Single-day model performance
                f.write("SINGLE-DAY MODEL PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                for target, target_metrics in metrics.items():
                    f.write(f"{TARGET_NAMES[target]:25} - R²: {target_metrics['R2']:.4f}, RMSE: {target_metrics['RMSE']:.4f}\n")
                
                f.write("\n")
                
                # Multi-day forecasting performance
                f.write("MULTI-DAY FORECASTING PERFORMANCE\n")
                f.write("-" * 35 + "\n")
                for target, target_results in multi_day_results.items():
                    f.write(f"\n{TARGET_NAMES[target]}:\n")
                    for day, day_results in target_results.items():
                        if 'metrics' in day_results:
                            metrics = day_results['metrics']
                            f.write(f"  Day {day}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}\n")
                
                f.write("\n")
                f.write("CONFIGURATION\n")
                f.write("-" * 15 + "\n")
                f.write(f"Target Variables: {', '.join(TARGET_COLUMNS)}\n")
                f.write(f"Feature Count: {len(FEATURE_COLUMNS)}\n")
                f.write(f"Forecast Horizon: {FORECAST_DAYS} days\n")
                f.write(f"Cross-validation Folds: {CV_FOLDS}\n")
                f.write(f"Parallel Jobs: {N_JOBS}\n")
            
            print(f"Final summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error generating final summary: {str(e)}")

def main():
    """Main execution function"""
    try:
        # Initialize the predictor
        predictor = GBMWeatherPredictor()
        
        # Run the complete pipeline
        results = predictor.run_complete_pipeline()
        
        print("\nProcess completed successfully.")
        return results
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        # Clean up matplotlib resources
        import matplotlib.pyplot as plt
        plt.close('all')

if __name__ == "__main__":
    main() 