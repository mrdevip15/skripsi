import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

def build_lstm_model(input_shape):
    """LSTM model with simpler architecture"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),  # Reduced from 128
        BatchNormalization(),
        
        LSTM(32),                                                  # Reduced from 64
        BatchNormalization(),
        
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
                n_estimators=500,          # Increased number of trees
                max_depth=12,              # Deeper trees
                min_samples_split=2,       # Allow for finer splits
                min_samples_leaf=1,        # Allow smaller leaf nodes
                max_features='sqrt',       # Use sqrt of features for each split
                bootstrap=True,            # Enable bootstrapping
                random_state=42,
                n_jobs=-1                  # Use all CPU cores
            ),
            'name': 'Random Forest',
            'type': 'sklearn'
        },
        {
            'model': build_lstm_model,
            'name': 'LSTM',
            'type': 'keras',
            'epochs': 100,         # Reduced from 200
            'batch_size': 32       # Reduced from 64
        },
        {
            'model': SVR(
                kernel='rbf',
                C=1.0,             # Reduced from 10.0
                epsilon=0.1,
                gamma='scale'
            ),
            'name': 'SVM',
            'type': 'sklearn'
        },
        {
            'model': GradientBoostingRegressor(
                n_estimators=100,   # Reduced from 200
                learning_rate=0.1,  # Increased from 0.05
                max_depth=4,        # Reduced from 5
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