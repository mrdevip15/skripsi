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
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.ensemble import GradientBoostingRegressor  # for GBM

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
        self.history = None
        self.config = CONFIG
        self.model_type = None
        self.model_name = None
        self.model_builder = None
        
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
            X.append(sequence.flatten())  # Flatten for model input
            y.append(df['RR'].values[i+lookback])
        
        # Store the sequence shape for prediction
        self.sequence_shape = (lookback, len(features))
        return np.array(X), np.array(y)

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

    def train(self, X, y):
        """Train the model with detailed evaluation"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            
            # Handle different model types
            if self.model_type == 'keras':
                # Reshape input for LSTM and CNN
                if self.model_name == 'LSTM':
                    X_train = X_train.reshape(X_train.shape[0], self.sequence_shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], self.sequence_shape[0], -1)
                elif self.model_name == 'CNN':
                    # Reshape for CNN (samples, timesteps, features)
                    X_train = X_train.reshape(X_train.shape[0], -1, 1)
                    X_test = X_test.reshape(X_test.shape[0], -1, 1)
                
                # Build and train model
                self.model = self.model_builder((X_train.shape[1:]))
                
                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # Make predictions
                y_train_pred = self.model.predict(X_train).flatten()
                y_test_pred = self.model.predict(X_test).flatten()
                
            else:  # sklearn models
                self.model.fit(X_train, y_train)
                y_train_pred = self.model.predict(X_train)
                y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.evaluate_predictions(y_train, y_train_pred)
            test_metrics = self.evaluate_predictions(y_test, y_test_pred)
            
            logging.info(f"\nModel: {self.model_name}")
            logging.info("\nTraining Metrics:")
            logging.info(f"Train MAE: {train_metrics['MAE']:.4f}")
            logging.info(f"Train R2: {train_metrics['R2']:.4f}")
            logging.info("\nTest Metrics:")
            logging.info(f"Test MAE: {test_metrics['MAE']:.4f}")
            logging.info(f"Test R2: {test_metrics['R2']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise

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
    5. Removing specific outlier values (e.g., RR = 8888 or 9999)
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
    
    # Remove rows where RR equals 8888 or 9999
    df_cleaned = df_cleaned[(df_cleaned['RR'] != 8888) & (df_cleaned['RR'] != 9999)]
    
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
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv('data_jakarta.csv')
        df_cleaned = preprocess_data(df)
        
        # Define model configurations
        configurations = [
            {
                'model': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=None, 
                    random_state=42
                ),
                'name': 'Random Forest',
                'type': 'sklearn'
            },
            {
                'model': build_lstm_model,
                'name': 'LSTM',
                'type': 'keras',
                'epochs': 100,
                'batch_size': 32
            },
            {
                'model': SVR(
                    kernel='rbf', 
                    C=1.0, 
                    epsilon=0.1
                ),
                'name': 'SVM',
                'type': 'sklearn'
            },
            {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                ),
                'name': 'GBM',
                'type': 'sklearn'
            },
            {
                'model': build_cnn_model,
                'name': 'CNN',
                'type': 'keras',
                'epochs': 100,
                'batch_size': 32
            },
            {
                'model': build_mlp_model,
                'name': 'Backpropagation (MLP)',
                'type': 'keras',
                'epochs': 100,
                'batch_size': 32
            }
        ]
        
        # Engineer features (without scaling)
        logging.info("Engineering features...")
        df_engineered = predictor.engineer_features(df_cleaned)
        
        # Prepare sequences
        logging.info("Preparing sequences...")
        X, y = predictor.prepare_sequences(df_engineered)
        
        if len(X) == 0:
            logging.error("No valid sequences could be created from the data")
            return
            
        best_model = None
        best_metrics = None
        best_mae = float('inf')
        
        # Create a list to store results for comparison
        results = []
        
        # Try each configuration
        for i, config in enumerate(configurations, 1):
            logging.info(f"\nTrying configuration {i}/{len(configurations)}: {config['name']}")
            
            # Update model configuration
            predictor.model_type = config['type']
            predictor.model_name = config['name']
            
            if config['type'] == 'keras':
                predictor.model_builder = config['model']
                predictor.config['epochs'] = config['epochs']
                predictor.config['batch_size'] = config['batch_size']
            else:
                predictor.model = config['model']
            
            # Train and evaluate
            try:
                metrics = predictor.train(X, y)
                current_mae = metrics['MAE']
                
                # Store results
                results.append({
                    'Model': config['name'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'MAPE': metrics['MAPE']
                })
                
                # Check if this is the best model so far
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_metrics = metrics
                    best_model = predictor.model
                    best_model_type = config['type']
                    best_model_name = config['name']
                    
                    # Save the best model
                    predictor.save_model()
                    logging.info(f"\nNew best model found! ({config['name']})")
                    logging.info(f"MAE: {current_mae:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {config['name']}: {str(e)}")
                results.append({
                    'Model': config['name'],
                    'MAE': None,
                    'RMSE': None,
                    'R2': None,
                    'MAPE': None
                })
                continue
        
        # Create and display comparison table
        logging.info("\n" + "="*50)
        logging.info("MODEL COMPARISON RESULTS")
        logging.info("="*50)
        
        # Convert results to DataFrame for better formatting
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('MAE')  # Sort by MAE
        
        # Format the table
        formatted_table = results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x) if pd.notnull(x) else 'Failed')
        logging.info("\n" + formatted_table)
        
        logging.info("\n" + "="*50)
        logging.info(f"BEST MODEL: {best_model_name}")
        logging.info("="*50)
        logging.info(f"MAE: {best_metrics['MAE']:.4f}")
        logging.info(f"RMSE: {best_metrics['RMSE']:.4f}")
        logging.info(f"R2 Score: {best_metrics['R2']:.4f}")
        logging.info(f"MAPE: {best_metrics['MAPE']:.4f}")
        
        # Use the best model for predictions
        if best_model is not None:
            predictor.model = best_model
            
            # Make predictions with best model
            last_sequence = X[-1].reshape(predictor.sequence_shape)
            predictions = predictor.predict_next_days(last_sequence)
            
            # Print predictions
            logging.info("\n" + "="*50)
            logging.info(f"PREDICTIONS USING BEST MODEL ({best_model_name})")
            logging.info("="*50)
            last_date = df_engineered['Tanggal'].iloc[-1]
            for i, pred in enumerate(predictions, 1):
                future_date = last_date + timedelta(days=i)
                logging.info(f"{future_date.date()}: {pred:.2f} mm")
        else:
            logging.error("No successful model training found")
                
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

# Define model builders for neural network models
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
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
