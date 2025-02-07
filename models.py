import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

def build_lstm_model(input_shape):
    """LSTM model with BatchNorm and proper dropout"""
    model = Sequential([
        # First LSTM layer
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(64),
        BatchNormalization(),
        
        # Dense layers with dropout
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_cnn_model(input_shape):
    """CNN model with BatchNorm and proper regularization"""
    if len(input_shape) == 1:
        input_shape = (input_shape[0], 1)
    
    model = Sequential([
        # First Conv block
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Second Conv block
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_mlp_model(input_shape):
    """MLP model with proper regularization"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape[0],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def get_model_configurations():
    """Return a list of model configurations"""
    return [
        {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'name': 'Random Forest',
            'type': 'sklearn'
        },
        {
            'model': build_lstm_model,
            'name': 'LSTM',
            'type': 'keras',
            'epochs': 200,
            'batch_size': 64
        },
        {
            'model': SVR(
                kernel='rbf',
                C=10.0,
                epsilon=0.1,
                gamma='scale'
            ),
            'name': 'SVM',
            'type': 'sklearn'
        },
        {
            'model': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'name': 'GBM',
            'type': 'sklearn'
        },
        {
            'model': build_cnn_model,
            'name': 'CNN',
            'type': 'keras',
            'epochs': 200,
            'batch_size': 64
        },
        {
            'model': build_mlp_model,
            'name': 'Backpropagation (MLP)',
            'type': 'keras',
            'epochs': 200,
            'batch_size': 64
        }
    ]