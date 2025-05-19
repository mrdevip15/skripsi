import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_latest_model():
    """Load the most recently trained model from the gbm/models directory"""
    models_dir = os.path.join('gbm', 'models')
    model_files = glob.glob(os.path.join(models_dir, 'gbm_model_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError("No trained models found. Please run main_gbm.py first.")
    
    # Sort by modification time (newest first)
    latest_model_file = max(model_files, key=os.path.getmtime)
    
    logging.info(f"Loading model: {latest_model_file}")
    
    with open(latest_model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def list_available_models():
    """List all available models with timestamps"""
    models_dir = os.path.join('gbm', 'models')
    model_files = glob.glob(os.path.join(models_dir, 'gbm_model_*.pkl'))
    
    if not model_files:
        return []
    
    models_info = []
    for model_file in model_files:
        try:
            # Extract timestamp from filename
            timestamp = os.path.basename(model_file).replace('gbm_model_', '').replace('.pkl', '')
            
            # Get file creation time
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            
            # Load model to get info
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models_info.append({
                'path': model_file,
                'timestamp': timestamp,
                'created': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                'features': len(model_data['feature_columns']),
                'target': model_data['target_column']
            })
        except Exception as e:
            logging.warning(f"Error loading model info from {model_file}: {str(e)}")
    
    return sorted(models_info, key=lambda x: x['created'], reverse=True)

def get_user_input(feature_names):
    """Get weather data input from user"""
    print("\nEnter weather data:")
    print("-" * 50)
    
    data = {}
    for feature in feature_names:
        if feature in ['Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range'] or \
           feature.startswith(('RR_Rolling', 'Tavg_Rolling', 'RR_Lag', 'Tavg_Lag')):
            # Skip derived features - these will be calculated
            continue
            
        while True:
            try:
                value = float(input(f"{feature}: "))
                data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return data

def prepare_input_data(user_input, model_data):
    """Prepare input data for prediction"""
    # Create a dataframe with a single row
    df = pd.DataFrame([user_input])
    
    # Add calculated features
    if 'Temp_Range' in model_data['feature_columns'] and 'Tn' in df and 'Tx' in df:
        df['Temp_Range'] = df['Tx'] - df['Tn']
    
    # Set default values for time-based features if they're needed
    today = datetime.now()
    if 'Month' in model_data['feature_columns']:
        df['Month'] = today.month
    if 'Day' in model_data['feature_columns']:
        df['Day'] = today.day
    if 'DayOfWeek' in model_data['feature_columns']:
        df['DayOfWeek'] = today.weekday()
    if 'Season' in model_data['feature_columns']:
        df['Season'] = (today.month % 12 + 3) // 3
    
    # For rolling and lag features, use median values from training data
    for feature in model_data['feature_columns']:
        if feature not in df.columns:
            if feature.startswith(('RR_Rolling', 'Tavg_Rolling', 'RR_Lag', 'Tavg_Lag')):
                # Use a default value of 0 for these features
                df[feature] = 0
    
    # Select only the features the model was trained on
    input_features = df[model_data['feature_columns']]
    
    # Scale the input data
    if 'scaler' in model_data:
        input_features = model_data['scaler'].transform(input_features)
    
    return input_features

def predict_rainfall(input_data, model):
    """Predict rainfall using the trained model"""
    prediction = model.predict(input_data)[0]
    return max(0, prediction)  # Rainfall can't be negative

def main():
    try:
        print("\nWeather Prediction Model - Rainfall Forecasting")
        print("=" * 50)
        
        # List available models
        models = list_available_models()
        if not models:
            print("No trained models found. Please run main_gbm.py first.")
            return
        
        print("\nAvailable models:")
        for i, model_info in enumerate(models, 1):
            print(f"{i}. Created: {model_info['created']} - Features: {model_info['features']}")
        
        # Model selection
        selected_idx = 0
        while True:
            try:
                selection = input("\nSelect model (press Enter for latest): ")
                if selection.strip() == '':
                    selected_idx = 0
                    break
                selected_idx = int(selection) - 1
                if 0 <= selected_idx < len(models):
                    break
                print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number.")
        
        # Load selected model
        model_path = models[selected_idx]['path']
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_columns']
        target_column = model_data['target_column']
        
        print(f"\nLoaded model created on {models[selected_idx]['created']}")
        print(f"This model predicts {target_column} using {len(feature_names)} features")
        
        # Get basic weather inputs from user
        basic_features = [f for f in feature_names if f in ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x']]
        user_input = get_user_input(basic_features)
        
        # Prepare input data
        input_data = prepare_input_data(user_input, model_data)
        
        # Make prediction
        prediction = predict_rainfall(input_data, model)
        
        # Display result
        print("\nPrediction Result:")
        print("=" * 50)
        print(f"Predicted Rainfall: {prediction:.2f} mm")
        
        # Interpret the prediction
        if prediction < 0.5:
            print("Interpretation: No significant rainfall expected")
        elif prediction < 5:
            print("Interpretation: Light rain expected")
        elif prediction < 20:
            print("Interpretation: Moderate rain expected")
        elif prediction < 50:
            print("Interpretation: Heavy rain expected")
        else:
            print("Interpretation: Very heavy rain expected")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 