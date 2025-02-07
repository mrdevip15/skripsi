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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.ensemble import GradientBoostingRegressor  # for GBM
from tensorflow.keras.optimizers import Adam
from models import get_model_configurations
from preprocessing import engineer_features
from visualization import (
    plot_model_comparison,
    plot_predictions,
    plot_feature_importance,
    plot_error_distribution
)

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
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_type = None
        self.model_name = None
        self.model_builder = None
        self.sequence_shape = None
        self.dates = None
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def engineer_features(self, df):
        """Add engineered features to the dataset"""
        df = df.copy()
        
        logging.info(f"Initial data shape: {df.shape}")
        
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
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        
        logging.info(f"Rows removed due to NaN after feature engineering: {removed_rows}")
        logging.info(f"Final data shape: {df.shape}")
        
        return df

    def prepare_sequences(self, df):
        """Prepare sequences for prediction"""
        features = self.config['INPUT_FEATURES']
        lookback = self.config['LOOKBACK_DAYS']
        
        # Check if all required features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logging.error(f"Missing features in data: {missing_features}")
            return np.array([]), np.array([]), np.array([])  # Added empty dates array
        
        logging.info(f"Data shape before sequence creation: {df.shape}")
        
        if len(df) <= lookback:
            logging.error(f"Not enough data points. Need more than {lookback} rows")
            return np.array([]), np.array([]), np.array([])  # Added empty dates array
        
        X, y, dates = [], [], []  # Added dates list
        for i in range(len(df) - lookback):
            sequence = df[features].values[i:i+lookback]
            if not np.isnan(sequence).any():
                X.append(sequence.flatten())
                y.append(df['RR'].values[i+lookback])
                dates.append(df['Tanggal'].values[i+lookback])  # Store corresponding date
        
        logging.info(f"Created {len(X)} sequences from {len(df)} data points")
        
        if len(X) == 0:
            logging.error("No valid sequences could be created")
            return np.array([]), np.array([]), np.array([])  # Added empty dates array
        
        self.sequence_shape = (lookback, len(features))
        return np.array(X), np.array(y), np.array(dates)  # Return dates array

    def build_model(self, input_shape):
        """Build model for single target (RR) prediction"""
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42,
            alpha=0.01  # L2 regularization
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

    def train_and_evaluate(self, X, y, dates, config):
        """Train and evaluate a model configuration"""
        from sklearn.model_selection import train_test_split
        
        # Split the data and dates together
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=self.config['TEST_SPLIT'], random_state=42
        )
        
        try:
            if config['type'] == 'keras':
                # Handle Keras models
                if config['name'] in ['LSTM', 'CNN']:
                    X_train = X_train.reshape(X_train.shape[0], self.sequence_shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], self.sequence_shape[0], -1)
                
                self.model = config['model'](X_train.shape[1:])
                
                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                y_pred = self.model.predict(X_test).flatten()
                
            else:
                # Handle sklearn models
                self.model = config['model']
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Create visualizations with actual dates
            plot_predictions(
                dates=dates_test,
                actual=y_test,
                predicted=y_pred,
                model_name=config['name'],
                save_path='plots'
            )
            
            plot_error_distribution(
                actual=y_test,
                predicted=y_pred,
                model_name=config['name'],
                save_path='plots'
            )
            
            if config['type'] == 'sklearn' and hasattr(self.model, 'feature_importances_'):
                plot_feature_importance(
                    model=self.model,
                    feature_names=self.config['INPUT_FEATURES'],
                    save_path='plots'
                )
            
            return metrics, self.model
            
        except Exception as e:
            logging.error(f"Error training {config['name']}: {str(e)}")
            return None, None

    def calculate_metrics(self, y_true, y_pred):
        """Calculate various performance metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

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
        logging.info(f"Model saved to {self.config['MODEL_SAVE_PATH']}")

    def load_model(self):
        """Load saved model and scaler"""
        try:
            import pickle
            with open(self.config['MODEL_SAVE_PATH'], 'rb') as f:
                self.model = pickle.load(f)
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
                
                # Prepare sequences
                X, y, dates = predictor.prepare_sequences(df_engineered)
                
                # Train and evaluate
                metrics, model = predictor.train_and_evaluate(X, y, dates, trial_config)
                mae_percentage = metrics['MAE_as_percentage_of_mean']
                
                # Check if this is the best model so far
                if mae_percentage < best_mae_percentage:
                    best_mae_percentage = mae_percentage
                    self.best_metrics = metrics
                    self.best_config = trial_config.copy()
                    self.best_model = model
                    
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
    if pd.isna(direction):
        return None
        
    direction_dict = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    
    # Clean the input string
    direction = str(direction).strip().upper()
    return direction_dict.get(direction, 0)

def preprocess_data(df):
    """
    Preprocess the weather data by:
    1. Converting wind directions to numerical values
    2. Converting all columns to numeric type
    3. Removing invalid values (8888, 9999)
    4. Normalizing features
    """
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Print initial data info
    logging.info(f"Initial data shape: {df.shape}")
    initial_rows = len(df)
    
    try:
        # Convert date column
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        
        # Convert wind directions to numerical values
        df['ddd_car'] = df['ddd_car'].str.strip()  # Remove any whitespace
        df['ddd_car'] = df['ddd_car'].apply(convert_wind_direction)
        
        # Convert all columns (except date) to numeric
        for column in df.columns:
            if column != 'Tanggal':
                df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Remove invalid rainfall values (8888, 9999)
        df = df[~df['RR'].isin([8888, 9999])]
        rows_after_invalid = len(df)
        
        # Remove rows with any missing values
        df = df.dropna()
        rows_after_nan = len(df)
        
        # Normalize numerical columns (except date and target)
        numerical_cols = [col for col in df.columns if col not in ['Tanggal', 'RR']]
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:  # Avoid division by zero
                df[col] = (df[col] - mean) / std
        
        # Log preprocessing results
        logging.info("\nPreprocessing Results:")
        logging.info(f"Initial rows: {initial_rows}")
        logging.info(f"Rows after removing invalid RR: {rows_after_invalid}")
        logging.info(f"Rows after removing NaN: {rows_after_nan}")
        logging.info(f"Final rows: {len(df)}")
        
        # Log RR statistics
        logging.info("\nRainfall (RR) Statistics:")
        logging.info(f"Mean: {df['RR'].mean():.2f}")
        logging.info(f"Std: {df['RR'].std():.2f}")
        logging.info(f"Min: {df['RR'].min():.2f}")
        logging.info(f"Max: {df['RR'].max():.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def main():
    try:
        # Initialize
        predictor = WeatherPredictor(CONFIG)
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv('data_jakarta.csv')
        df_cleaned = preprocess_data(df)
        
        # Engineer features
        df_engineered = predictor.engineer_features(df_cleaned)
        
        # Prepare sequences
        X, y, dates = predictor.prepare_sequences(df_engineered)
        
        if len(X) == 0:
            return
        
        # Get model configurations
        configurations = get_model_configurations()
        
        # Store results
        results = []
        best_model = None
        best_metrics = None
        best_mae = float('inf')
        
        # Train and evaluate each model
        for config in configurations:
            logging.info(f"\nTraining {config['name']}...")
            
            metrics, model = predictor.train_and_evaluate(X, y, dates, config)
            
            if metrics is not None:
                results.append({
                    'Model': config['name'],
                    **metrics
                })
                
                if metrics['MAE'] < best_mae:
                    best_mae = metrics['MAE']
                    best_metrics = metrics
                    best_model = model
                    best_model_name = config['name']
        
        # Create comparison plot
        results_df = pd.DataFrame(results)
        plot_model_comparison(results_df, save_path='plots')
        
        # Log results
        logging.info("\n" + "="*50)
        logging.info("MODEL COMPARISON RESULTS")
        logging.info("="*50)
        logging.info("\n" + results_df.to_string(index=False))
        
        logging.info("\n" + "="*50)
        logging.info(f"BEST MODEL: {best_model_name}")
        logging.info("="*50)
        for metric, value in best_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

# Define model builders for neural network models
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_cnn_model(input_shape):
    """Build CNN model with proper input shape handling"""
    # Reshape input_shape if needed
    if len(input_shape) == 1:
        # Add channel dimension for 1D CNN
        input_shape = (input_shape[0], 1)
    
    model = Sequential([
        # First Conv1D layer
        Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu',
            input_shape=input_shape,
            padding='same'
        ),
        MaxPooling1D(pool_size=2),
        
        # Second Conv1D layer
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        MaxPooling1D(pool_size=2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    # Compile model with MSE loss and Adam optimizer
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape[0],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    main()
