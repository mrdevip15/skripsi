"""Configuration settings for the weather prediction model"""

CONFIG = {
    # Model parameters
    'INPUT_FEATURES': ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
    'TARGET_FEATURE': 'PRCP',
    'HIDDEN_LAYERS': [64, 32],  # Multiple hidden layers
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 64,
    'EPOCHS': 200,
    'VALIDATION_SPLIT': 0.2,
    'TEST_SPLIT': 0.2,
    'EARLY_STOPPING_PATIENCE': 10,
    
    # Data parameters
    'LOOKBACK_DAYS': 7,  # Adjust as needed
    'PREDICTION_DAYS': 7,  # Number of days to predict ahead
    
    # File paths
    'MODEL_SAVE_PATH': 'models/weather_model.pkl',
    'PLOT_SAVE_PATH': 'plots/',
    
    # Feature engineering
    'USE_FEATURE_ENGINEERING': True,
    'ROLLING_WINDOW_SIZES': [3, 7, 14, 30],  # Added more window sizes
    'FEATURE_SELECTION_THRESHOLD': 0.01,  # For removing low-importance features
} 