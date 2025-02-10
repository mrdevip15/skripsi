import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svm_prediction.log'),
        logging.StreamHandler()
    ]
)

class SVMPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        df = df.copy()
        
        logging.info(f"Initial shape: {df.shape}")
        
        try:
            # Keep only required columns
            required_columns = ['Tanggal'] + self.config['FEATURES'] + ['RR']
            df = df[required_columns]
            
            # Convert date
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            
            # Convert to numeric
            for col in df.columns:
                if col != 'Tanggal':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid rainfall values
            df = df[~df['RR'].isin([8888, 9999])]
            logging.info(f"Rows after removing invalid RR: {len(df)}")
            
            # Remove high rainfall values
            df = df[df['RR'] <= 60]
            logging.info(f"Rows after filtering RR > 60: {len(df)}")
            
            # Handle missing values
            df = df.set_index('Tanggal')
            df = df.interpolate(method='time')
            df = df.reset_index()
            
            # Log statistics
            self._log_statistics(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def prepare_features(self, df):
        """Prepare features for training"""
        X = df[self.config['FEATURES']].values
        y = df['RR'].values
        dates = df['Tanggal'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y, dates
    
    def train_and_evaluate(self, X, y, dates):
        """Train and evaluate the SVM model"""
        # Split the data
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=self.config['TEST_SPLIT'], random_state=42
        )
        
        # Create and train the model
        self.model = SVR(
            kernel=self.config['KERNEL'],
            C=self.config['C'],
            epsilon=self.config['EPSILON'],
            gamma=self.config['GAMMA']
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Plot results
        self._plot_predictions(dates_test, y_test, y_pred)
        self._plot_error_distribution(y_test, y_pred)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def _log_statistics(self, df):
        """Log data statistics"""
        logging.info("\nFeature Statistics:")
        for col in self.config['FEATURES'] + ['RR']:
            stats = df[col].describe()
            logging.info(f"\n{col}:")
            logging.info(f"Mean: {stats['mean']:.2f}")
            logging.info(f"Std: {stats['std']:.2f}")
            logging.info(f"Min: {stats['min']:.2f}")
            logging.info(f"Max: {stats['max']:.2f}")
    
    def _plot_predictions(self, dates, y_true, y_pred):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15, 6))
        plt.plot(dates, y_true, label='Actual', marker='o')
        plt.plot(dates, y_pred, label='Predicted', marker='s')
        plt.title('Actual vs Predicted Rainfall - SVM')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/svm_predictions.png')
        plt.close()
    
    def _plot_error_distribution(self, y_true, y_pred):
        """Plot error distribution"""
        errors = y_pred - y_true
        
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.hist(errors, bins=30, density=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        
        plt.subplot(122)
        plt.scatter(y_true, y_pred)
        plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        plt.tight_layout()
        plt.savefig('plots/svm_error_distribution.png')
        plt.close()

def main():
    # Configuration
    config = {
        'FEATURES': ['Tavg', 'RH_avg', 'ss', 'ff_x', 'ddd_x'],  # Features to use
        'TEST_SPLIT': 0.2,
        'KERNEL': 'rbf',
        'C': 10.0,
        'EPSILON': 0.1,
        'GAMMA': 'scale'
    }
    
    try:
        # Initialize predictor
        predictor = SVMPredictor(config)
        
        # Load and preprocess data
        df = pd.read_csv('data_jakarta.csv')
        df_processed = predictor.preprocess_data(df)
        
        # Prepare features
        X, y, dates = predictor.prepare_features(df_processed)
        
        # Train and evaluate
        metrics = predictor.train_and_evaluate(X, y, dates)
        
        logging.info("\nTraining completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 