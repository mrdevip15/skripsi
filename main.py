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
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

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

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Single output for rainfall prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def prepare_sequences(self, df):
        """Prepare sequences for RR prediction only"""
        features = self.config['INPUT_FEATURES']
        lookback = self.config['LOOKBACK_DAYS']
        
        X, y = [], []
        for i in range(len(df) - lookback):
            X.append(df[features].values[i:i+lookback])
            y.append(df['RR'].values[i+lookback])
        
        return np.array(X), np.array(y)

    def train(self, X, y):
        """Train the model with early stopping and validation"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['TEST_SPLIT'],
            random_state=42
        )
        
        # Build model
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['EARLY_STOPPING_PATIENCE'],
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            self.config['MODEL_SAVE_PATH'],
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=self.config['VALIDATION_SPLIT'],
            epochs=self.config['EPOCHS'],
            batch_size=self.config['BATCH_SIZE'],
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = self.evaluate_predictions(y_test, y_pred.flatten())
        
        return metrics

    def predict_next_days(self, last_sequence):
        """Predict rainfall for next few days"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(self.config['PREDICTION_DAYS']):
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, self.config['INPUT_FEATURES'].index('RR')] = pred[0, 0]
        
        return np.array(predictions)

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_SAVE_PATH'], 'training_history.png'))
        plt.close()

    def save_model(self):
        """Save model and scaler"""
        self.model.save(self.config['MODEL_SAVE_PATH'])
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logging.info(f"Model saved to {self.config['MODEL_SAVE_PATH']}")

    def load_model(self):
        """Load saved model and scaler"""
        try:
            self.model = load_model(self.config['MODEL_SAVE_PATH'])
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
