"""Configuration settings for the weather prediction model"""

CONFIG = {
    # Model parameters
    'INPUT_FEATURES': ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'],
    'HIDDEN_LAYERS': [64, 32],  # Multiple hidden layers
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 64,
    'EPOCHS': 200,
    'VALIDATION_SPLIT': 0.2,
    'TEST_SPLIT': 0.2,
    'EARLY_STOPPING_PATIENCE': 10,
    
    # Data parameters
    'LOOKBACK_DAYS': 5,  # Reduced from 7
    'PREDICTION_DAYS': 5,  # Reduced from 7
    
    # File paths
    'MODEL_SAVE_PATH': 'models/weather_model.pkl',
    'PLOT_SAVE_PATH': 'plots/',
    
    # Feature engineering
    'USE_FEATURE_ENGINEERING': True,
    'ROLLING_WINDOW_SIZES': [3, 5],  # Reduced window sizes
} 