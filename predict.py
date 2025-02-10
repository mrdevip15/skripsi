import numpy as np
import logging
from config import CONFIG
from main import WeatherPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

def predict_with_manual_input(model, model_type='sklearn', predict_tomorrow=False):
    """
    Make predictions using manual input data
    predict_tomorrow: if True, predicts next day's precipitation
    """
    try:
        # Get input from user
        print("\nEnter the following weather data:")
        if predict_tomorrow:
            print("(Enter today's data to predict tomorrow's precipitation)")
        else:
            print("(Enter data to predict today's precipitation)")
            
        snow = float(input("Snow (mm): "))
        snwd = float(input("Snow Depth (mm): "))
        tmax = float(input("Maximum Temperature (°C): "))
        tmin = float(input("Minimum Temperature (°C): "))
        
        # Create input array
        input_data = np.array([0, snow, snwd, tmax, tmin])
        
        # Reshape input data based on model type
        if model_type == 'sklearn':
            input_data = input_data.reshape(1, -1)
        elif model_type == 'keras':
            input_data = input_data.reshape(1, 1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        if predict_tomorrow:
            print(f"\nPredicted Tomorrow's Precipitation: {prediction[0]:.2f} mm")
        else:
            print(f"\nPredicted Today's Precipitation: {prediction[0]:.2f} mm")
            
        return prediction[0]
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def predict_with_manual_input_batch(model, model_type='sklearn', num_predictions=3):
    """
    Make multiple predictions using manual input data
    """
    predictions = []
    
    try:
        for i in range(num_predictions):
            print(f"\nEnter data for prediction {i+1}:")
            snow = float(input("Snow (mm): "))
            snwd = float(input("Snow Depth (mm): "))
            tmax = float(input("Maximum Temperature (°C): "))
            tmin = float(input("Minimum Temperature (°C): "))
            
            input_data = np.array([0, snow, snwd, tmax, tmin])
            
            if model_type == 'sklearn':
                input_data = input_data.reshape(1, -1)
            elif model_type == 'keras':
                input_data = input_data.reshape(1, 1, -1)
            
            prediction = model.predict(input_data)
            predictions.append(prediction[0])
            
            print(f"Predicted Precipitation: {prediction[0]:.2f} mm")
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

def main():
    try:
        # Load the model
        predictor = WeatherPredictor(CONFIG)
        
        # List available model files
        print("\nAvailable models:")
        import os
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        for i, file in enumerate(model_files):
            print(f"{i+1}. {file}")
        
        # Let user choose model
        choice = int(input("\nChoose model number: ")) - 1
        model_path = f"models/{model_files[choice]}"
        
        model = predictor.load_specific_model(model_path)
        
        if model is not None:
            while True:
                print("\nChoose prediction type:")
                print("1. Today's precipitation")
                print("2. Tomorrow's precipitation")
                print("3. Multiple predictions")
                print("4. Exit")
                
                choice = input("Enter your choice (1-4): ")
                
                if choice == '1':
                    predict_with_manual_input(model, model_type='sklearn', predict_tomorrow=False)
                elif choice == '2':
                    predict_with_manual_input(model, model_type='sklearn', predict_tomorrow=True)
                elif choice == '3':
                    num = int(input("How many predictions? "))
                    predict_with_manual_input_batch(model, model_type='sklearn', num_predictions=num)
                elif choice == '4':
                    print("Exiting...")
                    break
                else:
                    print("Invalid choice. Please try again.")
        else:
            print("Failed to load model")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 