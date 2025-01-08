import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
from config import CONFIG
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_prediction.log'),
        logging.StreamHandler()
    ]
)

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.config = CONFIG
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def engineer_features(self, df):
        """Add engineered features to the dataset"""
        df = df.copy()
        
        if not self.config['USE_FEATURE_ENGINEERING']:
            return df
            
        # Add time-based features
        df['day_of_year'] = df['Tanggal'].dt.dayofyear
        df['month'] = df['Tanggal'].dt.month
        df['day_of_week'] = df['Tanggal'].dt.dayofweek
        
        # Add rolling statistics
        for window in self.config['ROLLING_WINDOW_SIZES']:
            for feature in self.config['INPUT_FEATURES']:
                df[f'{feature}_rolling_mean_{window}d'] = df[feature].rolling(window=window).mean()
                df[f'{feature}_rolling_std_{window}d'] = df[feature].rolling(window=window).std()
        
        # Drop rows with NaN values from rolling calculations
        df = df.dropna()
        
        return df

    def prepare_sequences(self, df):
        """Prepare sequences for RR prediction only"""
        features = self.config['INPUT_FEATURES']
        lookback = self.config['LOOKBACK_DAYS']
        
        X, y = [], []
        for i in range(len(df) - lookback):
            # Store the input shape for later use
            sequence = df[features].values[i:i+lookback]
            X.append(sequence.flatten())  # Flatten for MLPRegressor
            y.append(df['RR'].values[i+lookback])
        
        # Store the sequence shape for prediction
        self.sequence_shape = (lookback, len(features))
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build model for single target (RR) prediction"""
        model = MLPRegressor(
            hidden_layer_sizes=self.config['HIDDEN_LAYERS'],
            learning_rate_init=self.config['LEARNING_RATE'],
            max_iter=self.config['EPOCHS'],
            early_stopping=True,
            validation_fraction=self.config['VALIDATION_SPLIT'],
            n_iter_no_change=self.config['EARLY_STOPPING_PATIENCE'],
            random_state=42
        )
        return model

    def evaluate_predictions(self, y_true, y_pred):
        """Calculate various accuracy metrics with context"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Add context metrics
        mean_rainfall = np.mean(y_true)
        max_rainfall = np.max(y_true)
        mae_percentage = (mae / mean_rainfall) * 100  # MAE as percentage of mean rainfall
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'MAE_as_percentage_of_mean': mae_percentage,
            'Context': {
                'Mean_Rainfall': mean_rainfall,
                'Max_Rainfall': max_rainfall
            }
        }

    def train(self, X, y):
        """Train the model with detailed evaluation"""
        # Split data (X is already in the correct shape)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['TEST_SPLIT'],
            random_state=42
        )
        
        # Train model
        self.model = self.build_model(X.shape[1:])
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.evaluate_predictions(y_train, y_train_pred)
        test_metrics = self.evaluate_predictions(y_test, y_test_pred)
        
        return test_metrics

    def predict_next_days(self, last_sequence):
        """Predict rainfall for the next few days"""
        predictions = []
        # Ensure the input sequence has the correct shape
        current_sequence = last_sequence.reshape(self.sequence_shape).flatten()
        
        for _ in range(self.config['PREDICTION_DAYS']):
            # Reshape sequence for prediction
            reshaped_sequence = current_sequence.reshape(1, -1)
            pred = self.model.predict(reshaped_sequence)
            predictions.append(pred[0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -len(self.config['INPUT_FEATURES']))
            # Update only the RR value in the appropriate position
            rr_idx = self.config['INPUT_FEATURES'].index('RR')
            current_sequence[-len(self.config['INPUT_FEATURES']) + rr_idx] = pred[0]
        
        return np.array(predictions)

    def save_model(self):
        """Save model and scaler"""
        import pickle
        with open(self.config['MODEL_SAVE_PATH'], 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logging.info(f"Model saved to {self.config['MODEL_SAVE_PATH']}")

    def load_model(self):
        """Load saved model and scaler"""
        try:
            import pickle
            with open(self.config['MODEL_SAVE_PATH'], 'rb') as f:
                self.model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logging.info("Model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False

class ModelOptimizer:
    def __init__(self, max_trials=10):
        self.max_trials = max_trials
        self.best_model = None
        self.best_metrics = None
        self.best_config = None
        
    def generate_config(self, trial):
        """Generate a new configuration based on trial number"""
        if trial == 0:
            return {
                'HIDDEN_LAYERS': [64, 32],
                'LEARNING_RATE': 0.001,
                'LOOKBACK_DAYS': 5,
            }
        
        # Generate random configurations
        hidden_layers = random.choice([
            [32, 16],
            [64, 32],
            [128, 64],
            [64, 32, 16],
            [128, 64, 32],
            [256, 128, 64],
            [512, 256, 128],
            [1024, 512, 256],
            [64, 64, 64, 64],
            [128, 128, 128, 64],
            [256, 256, 128, 64],
        ])
        
        learning_rate = random.choice([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
        lookback_days = random.choice([3, 5, 7, 10, 14, 21, 30, 45, 60])
        
        return {
            'HIDDEN_LAYERS': hidden_layers,
            'LEARNING_RATE': learning_rate,
            'LOOKBACK_DAYS': lookback_days,
        }
    
    def optimize(self, predictor, df_cleaned):
        """Find the best model configuration"""
        best_mae_percentage = float('inf')
        
        for trial in range(self.max_trials):
            # Generate new configuration
            trial_config = self.generate_config(trial)
            
            # Update predictor config
            predictor.config.update(trial_config)
            
            logging.info(f"\nTrial {trial + 1}/{self.max_trials}")
            logging.info(f"Configuration: {trial_config}")
            
            try:
                # Engineer features
                df_engineered = predictor.engineer_features(df_cleaned)
                
                # Scale features
                features = predictor.config['INPUT_FEATURES']
                df_engineered[features] = predictor.scaler.fit_transform(df_engineered[features])
                
                # Prepare sequences
                X, y = predictor.prepare_sequences(df_engineered)
                
                # Train and evaluate
                metrics = predictor.train(X, y)
                mae_percentage = metrics['MAE_as_percentage_of_mean']
                
                # Check if this is the best model so far
                if mae_percentage < best_mae_percentage:
                    best_mae_percentage = mae_percentage
                    self.best_metrics = metrics
                    self.best_config = trial_config.copy()
                    self.best_model = predictor.model
                    
                    # Save the best model
                    predictor.save_model()
                    
                    logging.info(f"\nNew best model found!")
                    logging.info(f"MAE percentage: {mae_percentage:.2f}%")
                
                # Check if we've reached our goal
                if mae_percentage < 50:
                    logging.info(f"\nGoal achieved! MAE is less than 50% of mean rainfall")
                    return True
                
            except Exception as e:
                logging.error(f"Error in trial {trial + 1}: {str(e)}")
                continue
        
        return False

def prepare_data(df, lookback=3):
    # Prepare features and target
    features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    
    # Create sequences for the lookback period
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[features].values[i:i+lookback].flatten())
        y.append(df[features].values[i+lookback])
    
    return np.array(X), np.array(y)

def convert_wind_direction(direction):
    """Convert wind direction text to degrees"""
    direction_dict = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    return direction_dict.get(direction, 0)  # Returns 0 if direction not found

def preprocess_data(df):
    """
    Preprocess the weather data by:
    1. Removing rows with missing values
    2. Converting wind directions to numerical values
    3. Converting all columns to numeric type
    4. Handling outliers
    """
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Convert date column
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
    
    # List of features we want to keep
    features = ['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    
    # Keep only the columns we need
    df = df[features]
    
    # Convert wind directions to numerical values
    if 'ddd_x' in df.columns:
        df['ddd_x'] = df['ddd_x'].apply(convert_wind_direction)
    if 'ddd_car' in df.columns:
        df['ddd_car'] = df['ddd_car'].apply(convert_wind_direction)
    
    # Convert all columns (except date) to numeric, replacing errors with NaN
    for column in df.columns:
        if column != 'Tanggal':
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Remove rows with any missing values
    df_cleaned = df.dropna()
    
    # Print information about removed data
    total_rows = len(df)
    removed_rows = total_rows - len(df_cleaned)
    print(f"Total rows in original dataset: {total_rows}")
    print(f"Rows removed due to missing values: {removed_rows}")
    print(f"Remaining rows: {len(df_cleaned)}")
    
    return df_cleaned

def main():
    try:
        predictor = WeatherPredictor()
        optimizer = ModelOptimizer(max_trials=20)
        
        # Load and preprocess data
        df = pd.read_csv('weather_data.csv')
        df_cleaned = preprocess_data(df)
        
        # Run optimization
        success = optimizer.optimize(predictor, df_cleaned)
        
        if success:
            logging.info("\nOptimization successful!")
        else:
            logging.info("\nOptimization completed without reaching target.")
            logging.info("Using best found configuration:")
        
        logging.info("\nBest Configuration:")
        for key, value in optimizer.best_config.items():
            logging.info(f"{key}: {value}")
        
        logging.info("\nBest Metrics:")
        for metric, value in optimizer.best_metrics.items():
            if metric == 'Context':
                logging.info("\nContext Information:")
                for context_metric, context_value in value.items():
                    logging.info(f"{context_metric}: {context_value:.4f} mm")
            else:
                logging.info(f"{metric}: {value:.4f}")
        
        # Update predictor with best configuration before making predictions
        predictor.config.update(optimizer.best_config)
        predictor.model = optimizer.best_model
        
        # Engineer features and scale data again using best configuration
        df_engineered = predictor.engineer_features(df_cleaned)
        features = predictor.config['INPUT_FEATURES']
        df_engineered[features] = predictor.scaler.fit_transform(df_engineered[features])
        
        # Get the last sequence with correct shape using best configuration
        X, _ = predictor.prepare_sequences(df_engineered)
        last_sequence = X[-1].reshape(predictor.sequence_shape)
        predictions = predictor.predict_next_days(last_sequence)
        
        # Inverse transform predictions
        rr_idx = predictor.config['INPUT_FEATURES'].index('RR')
        rr_min = predictor.scaler.data_min_[rr_idx]
        rr_max = predictor.scaler.data_max_[rr_idx]
        predictions_original = predictions * (rr_max - rr_min) + rr_min
        
        # Print predictions
        last_date = df_engineered['Tanggal'].iloc[-1]
        print("\nRainfall (RR) Predictions using best model:")
        for i, pred in enumerate(predictions_original, 1):
            future_date = last_date + timedelta(days=i)
            print(f"{future_date.date()}: {pred:.2f} mm")
                
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
