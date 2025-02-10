import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import timedelta

def plot_model_comparison(results_df, save_path=None):
    """Plot comparison of model performances"""
    plt.figure(figsize=(12, 6))
    
    # Create bar plot for MAE
    plt.subplot(121)
    sns.barplot(x='Model', y='MAE', data=results_df)
    plt.xticks(rotation=45)
    plt.title('Model Comparison - MAE')
    
    # Create bar plot for R2
    plt.subplot(122)
    sns.barplot(x='Model', y='R2', data=results_df)
    plt.xticks(rotation=45)
    plt.title('Model Comparison - R2 Score')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/model_comparison.png")
    plt.close()

def plot_predictions(dates, actual, predicted, model_name, save_path):
    """Plot actual vs predicted values for test data only"""
    plt.figure(figsize=(12, 6))
    
    # Convert dates to datetime if they're not already
    dates = pd.to_datetime(dates)
    
    # Plot actual and predicted values
    plt.plot(dates, actual, label='Actual', color='blue', alpha=0.5)
    plt.plot(dates, predicted, label='Predicted', color='red', alpha=0.5)
    
    plt.title(f'Actual vs Predicted Values - {model_name} (Test Set Only)')
    plt.xlabel('Date')
    plt.ylabel('Precipitation')
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_path}/predictions_{model_name}.png')
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importance, y=feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/feature_importance.png")
        plt.close()

def plot_error_distribution(actual, predicted, model_name, save_path=None):
    """Plot error distribution"""
    errors = predicted - actual
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of errors
    plt.subplot(121)
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Prediction Error')
    
    # Q-Q plot
    plt.subplot(122)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/error_distribution_{model_name.lower().replace(' ', '_')}.png")
    plt.close() 