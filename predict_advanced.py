import os
import numpy as np
import pandas as pd
import joblib
import logging
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_latest_model():
    """Load the most recently trained advanced model"""
    models_dir = os.path.join('gbm_advanced', 'models')
    model_files = glob.glob(os.path.join(models_dir, 'advanced_gbm_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError("No advanced models found. Please run main_gbm_advanced.py first.")
    
    # Sort by modification time (newest first)
    latest_model_file = max(model_files, key=os.path.getmtime)
    
    logging.info(f"Loading model: {latest_model_file}")
    
    model_data = joblib.load(latest_model_file)
    
    return model_data

def list_available_models():
    """List all available advanced models with timestamps"""
    models_dir = os.path.join('gbm_advanced', 'models')
    model_files = glob.glob(os.path.join(models_dir, 'advanced_gbm_*.pkl'))
    
    if not model_files:
        return []
    
    models_info = []
    for model_file in model_files:
        try:
            # Extract timestamp from filename
            timestamp = os.path.basename(model_file).replace('advanced_gbm_', '').replace('.pkl', '')
            
            # Get file creation time
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            
            # Load model to get info
            model_data = joblib.load(model_file)
            
            models_info.append({
                'path': model_file,
                'timestamp': timestamp,
                'created': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                'features': len(model_data['feature_columns']),
                'forecast_days': model_data['forecast_days']
            })
        except Exception as e:
            logging.warning(f"Error loading model info from {model_file}: {str(e)}")
    
    return sorted(models_info, key=lambda x: x['created'], reverse=True)

def get_user_input(feature_names):
    """Get weather data input from user"""
    print("\nEnter current weather data:")
    print("-" * 50)
    
    data = {}
    for feature in feature_names:
        # Skip derived features - these will be calculated
        if feature in ['Month', 'Day', 'DayOfWeek', 'Season', 'Temp_Range', 
                       'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'] or \
           feature.startswith(('RR_Rolling', 'Tavg_Rolling', 'RH_Rolling', 'RR_Lag', 'Tavg_Lag', 'RH_Lag', 'RR_MA')):
            continue
            
        while True:
            try:
                value = float(input(f"{feature}: "))
                data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return data

def prepare_input_data(user_input, model_data, last_n_days=None):
    """Prepare input data for prediction including derived features"""
    # Create a DataFrame with current input
    today = pd.DataFrame([user_input])
    
    # Add date features
    current_date = datetime.now()
    
    # Add time features
    today['Month'] = current_date.month
    today['Day'] = current_date.day
    today['DayOfWeek'] = current_date.weekday()
    today['Season'] = (current_date.month % 12 + 3) // 3
    
    # Add Temperature Range if available
    if 'Tn' in today.columns and 'Tx' in today.columns:
        today['Temp_Range'] = today['Tx'] - today['Tn']
    
    # Add cyclical encoding
    today['Month_Sin'] = np.sin(2 * np.pi * today['Month']/12)
    today['Month_Cos'] = np.cos(2 * np.pi * today['Month']/12)
    today['DayOfWeek_Sin'] = np.sin(2 * np.pi * today['DayOfWeek']/7)
    today['DayOfWeek_Cos'] = np.cos(2 * np.pi * today['DayOfWeek']/7)
    
    # If historical data is provided, add rolling and lag features
    if last_n_days is not None:
        # Combine with historical data
        combined_data = pd.concat([last_n_days, today], ignore_index=True)
        
        # Add rolling features
        for window in [3, 7, 14, 30]:
            # Only add if we have enough historical data
            if len(combined_data) >= window:
                # Rainfall rolling stats
                combined_data[f'RR_Rolling_Mean_{window}d'] = combined_data['RR'].rolling(window=window, min_periods=1).mean()
                combined_data[f'RR_Rolling_Std_{window}d'] = combined_data['RR'].rolling(window=window, min_periods=1).std()
                combined_data[f'RR_Rolling_Max_{window}d'] = combined_data['RR'].rolling(window=window, min_periods=1).max()
                
                # Temperature rolling stats
                combined_data[f'Tavg_Rolling_Mean_{window}d'] = combined_data['Tavg'].rolling(window=window, min_periods=1).mean()
                combined_data[f'Tavg_Rolling_Std_{window}d'] = combined_data['Tavg'].rolling(window=window, min_periods=1).std()
                
                # Humidity rolling stats
                combined_data[f'RH_Rolling_Mean_{window}d'] = combined_data['RH_avg'].rolling(window=window, min_periods=1).mean()
        
        # Add lag features
        forecast_days = model_data['forecast_days']
        max_lag = forecast_days + 10
        for lag in range(1, max_lag + 1):
            if len(combined_data) > lag:
                combined_data[f'RR_Lag_{lag}'] = combined_data['RR'].shift(lag)
                combined_data[f'Tavg_Lag_{lag}'] = combined_data['Tavg'].shift(lag)
                combined_data[f'RH_Lag_{lag}'] = combined_data['RH_avg'].shift(lag)
        
        # Moving average differences
        if 'RR_Rolling_Mean_7d' in combined_data.columns and 'RR_Rolling_Mean_14d' in combined_data.columns:
            combined_data['RR_MA7_MA14_Diff'] = combined_data['RR_Rolling_Mean_7d'] - combined_data['RR_Rolling_Mean_14d']
        if 'RR_Rolling_Mean_3d' in combined_data.columns and 'RR_Rolling_Mean_7d' in combined_data.columns:
            combined_data['RR_MA3_MA7_Diff'] = combined_data['RR_Rolling_Mean_3d'] - combined_data['RR_Rolling_Mean_7d']
        
        # Extract the last row (today) with all features
        today = combined_data.iloc[-1:].copy()
    
    # Select only the features the model was trained on and fill missing with 0
    input_features = pd.DataFrame(index=today.index)
    for feature in model_data['feature_columns']:
        if feature in today.columns:
            input_features[feature] = today[feature]
        else:
            input_features[feature] = 0
    
    # Scale the input
    X = model_data['scaler'].transform(input_features)
    
    return X

def make_prediction(model_data, X):
    """Make predictions for multiple days ahead"""
    forecast_days = model_data['forecast_days']
    predictions = {}
    confidence_intervals = {}
    
    for day in range(1, forecast_days + 1):
        model_key = f'day_{day}'
        model = model_data['models'][model_key]
        
        # Make prediction
        pred = model.predict(X)[0]
        predictions[model_key] = max(0, pred)  # Rainfall can't be negative
        
        # Calculate confidence interval
        if 'prediction_stds' in model_data:
            std_dev = model_data['prediction_stds'][model_key][0]
            confidence_intervals[model_key] = {
                'lower': max(0, pred - 1.96 * std_dev),
                'upper': pred + 1.96 * std_dev
            }
    
    return predictions, confidence_intervals

def load_last_n_days(n=30):
    """Try to load recent historical data from a CSV file"""
    try:
        df = pd.read_csv('makassar.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        df = df.sort_values('Tanggal')
        
        # Convert any 8888 values in RR to NaN
        df['RR'] = df['RR'].replace(8888, np.nan)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Return the last n days
        return df.tail(n)
    except Exception as e:
        logging.warning(f"Could not load historical data: {e}")
        return None

def plot_forecast(predictions, confidence_intervals):
    """Plot the rainfall forecast for multiple days"""
    # Get dates for the forecast days
    dates = [datetime.now() + timedelta(days=i) for i in range(1, len(predictions) + 1)]
    date_labels = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Extract prediction values and intervals
    pred_values = [predictions[f'day_{i+1}'] for i in range(len(predictions))]
    
    plt.figure(figsize=(10, 6))
    
    # Plot predictions
    plt.bar(date_labels, pred_values, color='skyblue', alpha=0.7)
    
    # Add confidence intervals if available
    if confidence_intervals:
        lower_bounds = [confidence_intervals[f'day_{i+1}']['lower'] for i in range(len(predictions))]
        upper_bounds = [confidence_intervals[f'day_{i+1}']['upper'] for i in range(len(predictions))]
        
        plt.errorbar(date_labels, pred_values, 
                     yerr=[np.array(pred_values) - np.array(lower_bounds), 
                           np.array(upper_bounds) - np.array(pred_values)],
                     fmt='o', color='red', alpha=0.6)
    
    plt.title('Rainfall Forecast')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add colored background based on rainfall intensity
    for i, val in enumerate(pred_values):
        if val < 0.5:  # No rain
            color = 'lightyellow'
            category = 'No significant rainfall'
        elif val < 5:  # Light rain
            color = 'lightblue'
            category = 'Light rain'
        elif val < 20:  # Moderate rain
            color = 'skyblue'
            category = 'Moderate rain'
        elif val < 50:  # Heavy rain
            color = 'royalblue'
            category = 'Heavy rain'
        else:  # Very heavy rain
            color = 'navy'
            category = 'Very heavy rain'
            
        plt.annotate(f'{val:.1f} mm\n({category})', 
                     xy=(i, val), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', 
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.3))
    
    plt.tight_layout()
    
    # Save and show the plot
    os.makedirs('forecast_plots', exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    plt.savefig(f'forecast_plots/forecast_{today}.png')
    plt.show()
    
    return f'forecast_plots/forecast_{today}.png'

def main():
    try:
        print("\nAdvanced Weather Prediction - Multi-day Rainfall Forecast")
        print("=" * 60)
        
        # List available models
        models = list_available_models()
        if not models:
            print("No advanced models found. Please run main_gbm_advanced.py first.")
            return
        
        print("\nAvailable models:")
        for i, model_info in enumerate(models, 1):
            print(f"{i}. Created: {model_info['created']} - Features: {model_info['features']} - Forecast days: {model_info['forecast_days']}")
        
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
        model_data = joblib.load(model_path)
        
        forecast_days = model_data['forecast_days']
        feature_names = model_data['feature_columns']
        
        print(f"\nLoaded model created on {models[selected_idx]['created']}")
        print(f"This model can predict rainfall for the next {forecast_days} days using {len(feature_names)} features")
        
        # Try to load historical data
        historical_data = load_last_n_days(30)
        if historical_data is not None:
            print(f"\nLoaded historical data from the last {len(historical_data)} days to improve prediction accuracy")
        else:
            print("\nNo historical data available. Prediction accuracy may be reduced.")
        
        # Get current weather inputs from user
        basic_features = [f for f in feature_names if f in ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ff_avg', 'ddd_x']]
        user_input = get_user_input(basic_features)
        
        # Add rainfall value (needed for lag features)
        if 'RR' not in user_input and 'RR' in historical_data.columns:
            # Ask user for today's rainfall if known
            try:
                rainfall = float(input("\nEnter today's rainfall (mm) if known, or press Enter to use 0: "))
                user_input['RR'] = rainfall
            except ValueError:
                user_input['RR'] = 0
        elif 'RR' not in user_input:
            user_input['RR'] = 0
        
        # Prepare input data
        X = prepare_input_data(user_input, model_data, historical_data)
        
        # Make prediction
        predictions, confidence_intervals = make_prediction(model_data, X)
        
        # Display results
        print("\nRainfall Forecast:")
        print("=" * 60)
        
        for day in range(1, forecast_days + 1):
            forecast_date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            predicted_rain = predictions[f'day_{day}']
            
            confidence_text = ""
            if confidence_intervals:
                lower = confidence_intervals[f'day_{day}']['lower']
                upper = confidence_intervals[f'day_{day}']['upper']
                confidence_text = f" (95% confidence interval: {lower:.2f} - {upper:.2f} mm)"
            
            print(f"Day {day} ({forecast_date}): {predicted_rain:.2f} mm{confidence_text}")
            
            # Add interpretation
            if predicted_rain < 0.5:
                print("   Interpretation: No significant rainfall expected")
            elif predicted_rain < 5:
                print("   Interpretation: Light rain expected")
            elif predicted_rain < 20:
                print("   Interpretation: Moderate rain expected")
            elif predicted_rain < 50:
                print("   Interpretation: Heavy rain expected")
            else:
                print("   Interpretation: Very heavy rain expected")
        
        # Plot the forecast
        plot_path = plot_forecast(predictions, confidence_intervals)
        print(f"\nForecast plot saved to: {plot_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 