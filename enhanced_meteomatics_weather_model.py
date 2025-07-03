import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
from tqdm import tqdm
import time
import meteomatics.api as api

# Configure parallel processing
import multiprocessing
N_JOBS = max(1, multiprocessing.cpu_count() // 2)

# Suppress warnings
warnings.filterwarnings('ignore')

class EnhancedMeteomaticsWeatherPredictor:
    def __init__(self):
        # REMOVED WIND DIRECTION (ddd_car) from targets - poor performance
        self.target_columns = ['RR', 'ss', 'Tavg', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # Makassar coordinates for Meteomatics API
        self.latitude = -5.1477  # Makassar, Indonesia
        self.longitude = 119.4327
        
        # Meteomatics credentials
        self.username = "digiserv_hidayat_muhammad"
        self.password = "4kXR79jdKo"
        
        # Enhanced feature columns (excluding target variables)
        self.base_feature_columns = [
            # Original non-target features
            'Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x',
            
            # Temporal features
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos', 'Season',
            
            # Basic derived features
            'Temp_Range', 'Temp_Humidity', 'Dew_Point',
            
            # Lag features (from non-target variables only)
            'Tn_Lag_1', 'Tx_Lag_1', 'RH_avg_Lag_1',
            'Tn_Lag_2', 'Tx_Lag_2', 'RH_avg_Lag_2',
            
            # Rolling features (from non-target variables only)
            'Tn_Rolling_Mean_3d', 'Tx_Rolling_Mean_3d', 'RH_Rolling_Mean_3d',
            'Tn_Rolling_Std_3d', 'Tx_Rolling_Std_3d', 'RH_Rolling_Std_3d'
        ]
        
        # Will be populated with Meteomatics features
        self.meteomatics_features = []
        self.feature_columns = []
        
        self.scalers = {}
        self.models = {}
        self.cv_results = {}
        
        # Create directories
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'enhanced_meteomatics_results_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✅ Enhanced model initialized - Removed wind direction from targets")
        print(f"📍 Location: Makassar, Indonesia ({self.latitude}°, {self.longitude}°)")
        print(f"🎯 Target variables: {list(self.target_names.keys())}")
        print(f"📁 Results will be saved to: {self.results_dir}")

    def fetch_meteomatics_data(self, start_date, end_date):
        """Fetch additional meteorological data from Meteomatics API"""
        print("\n🌐 Fetching enhanced data from Meteomatics API...")
        
        # Define parameters to fetch (focusing on missing crucial variables)
        parameters = [
            # Atmospheric pressure (most important missing variable)
            'msl_pressure:hPa',        # Mean sea level pressure
            'sfc_pressure:hPa',        # Surface pressure
            
            # Enhanced humidity/atmospheric data
            'dewpoint_2m:C',           # Dew point temperature
            'precip_1h:mm',            # Hourly precipitation
            
            # Enhanced wind data
            'wind_speed_10m:ms',       # Wind speed at 10m
            'wind_dir_10m:d',          # Wind direction at 10m
            'wind_gusts_10m_1h:ms',    # Wind gusts
            
            # Cloud and visibility
            'total_cloud_cover:p',     # Total cloud cover percentage
            'visibility:km',           # Visibility
            
            # Additional meteorological parameters
            'relative_humidity_2m:p',  # Enhanced humidity
            'cape_ml:Jkg',            # Convective Available Potential Energy
        ]
        
        try:
            # Convert dates to datetime objects
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            print(f"📅 Fetching data from {start_date.date()} to {end_date.date()}")
            print(f"📊 Parameters: {len(parameters)} meteorological variables")
            
            # Create coordinate list
            coordinates = [(self.latitude, self.longitude)]
            
            # Fetch data using Meteomatics API
            meteomatics_df = api.query_time_series(
                startdate=start_date,
                enddate=end_date,
                interval=timedelta(days=1),  # Daily data
                parameters=parameters,
                coordinates=coordinates,
                username=self.username,
                password=self.password
            )
            
            print(f"✅ Successfully fetched Meteomatics data: {meteomatics_df.shape}")
            
            # Clean column names
            meteomatics_df.columns = [col.split(':')[0] for col in meteomatics_df.columns]
            
            # Add prefix to avoid name conflicts
            new_columns = {}
            for col in meteomatics_df.columns:
                if col != 'validdate':
                    new_columns[col] = f'meteo_{col}'
            meteomatics_df.rename(columns=new_columns, inplace=True)
            
            # Reset index and rename date column
            meteomatics_df.reset_index(inplace=True)
            meteomatics_df.rename(columns={'validdate': 'Date_API'}, inplace=True)
            
            # Store feature names for later use
            self.meteomatics_features = [col for col in meteomatics_df.columns if col.startswith('meteo_')]
            
            print(f"🔧 Added Meteomatics features: {len(self.meteomatics_features)}")
            
            return meteomatics_df
            
        except Exception as e:
            print(f"⚠️ Error fetching Meteomatics data: {str(e)}")
            print("📝 Continuing with original data only...")
            return None

    def preprocess_data(self, df):
        """Enhanced preprocessing with Meteomatics data integration"""
        print("\n🔧 Starting enhanced data preprocessing...")
        
        df = df.copy()
        initial_size = len(df)
        
        # Convert date column
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
        
        # Handle special values (8888, 9999) with interpolation
        special_values = [8888, 9999]
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                if any(df[col].isin(special_values)):
                    print(f"🔄 Interpolating special values in {col}")
                    mask = df[col].isin(special_values)
                    df.loc[mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        # Convert wind direction to numeric (but not using as target anymore)
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        if 'ddd_car' in df.columns:
            df['ddd_car_numeric'] = df['ddd_car'].map(wind_dir_map)
            df['ddd_car_numeric'] = df['ddd_car_numeric'].fillna(0)
        
        # Ensure target variables are numeric
        for target in self.target_columns:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors='coerce')
        
        # Add temporal features
        df['Month'] = df['Tanggal'].dt.month
        df['Day'] = df['Tanggal'].dt.day
        df['DayOfYear'] = df['Tanggal'].dt.dayofyear
        df['Season'] = (df['Month'] % 12 + 3) // 3
        
        # Cyclical encoding for seasonality
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day']/31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day']/31)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365.25)
        
        # Basic derived features
        df['Temp_Range'] = df['Tx'] - df['Tn']
        df['Temp_Humidity'] = ((df['Tx'] + df['Tn']) / 2) * df['RH_avg']
        df['Dew_Point'] = ((df['Tx'] + df['Tn']) / 2) - ((100 - df['RH_avg']) / 5)
        
        # Lag features (ONLY from non-target variables)
        non_target_vars = ['Tn', 'Tx', 'RH_avg']
        for var in non_target_vars:
            if var in df.columns:
                for lag in [1, 2]:
                    df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
        
        # Rolling features (ONLY from non-target variables)
        for var in non_target_vars:
            if var in df.columns:
                df[f'{var}_Rolling_Mean_3d'] = df[var].rolling(window=3, min_periods=1).mean()
                df[f'{var}_Rolling_Std_3d'] = df[var].rolling(window=3, min_periods=1).std()
        
        # Try to fetch Meteomatics data
        start_date = df['Tanggal'].min()
        end_date = df['Tanggal'].max()
        
        meteomatics_data = self.fetch_meteomatics_data(start_date, end_date)
        
        if meteomatics_data is not None:
            # Merge Meteomatics data
            meteomatics_data['Date_API'] = pd.to_datetime(meteomatics_data['Date_API']).dt.date
            df['Date_merge'] = df['Tanggal'].dt.date
            
            df = pd.merge(df, meteomatics_data, left_on='Date_merge', right_on='Date_API', how='left')
            df.drop(['Date_merge', 'Date_API'], axis=1, inplace=True)
            
            print(f"✅ Successfully merged Meteomatics data")
            
            # Handle missing values in Meteomatics data
            for col in self.meteomatics_features:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear')
                    df[col] = df[col].fillna(df[col].median())
        
        # Finalize feature columns
        available_base_features = [f for f in self.base_feature_columns if f in df.columns]
        available_meteomatics_features = [f for f in self.meteomatics_features if f in df.columns]
        
        self.feature_columns = available_base_features + available_meteomatics_features
        
        # Handle any remaining NaN values
        for col in self.feature_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Drop rows with NaN in target variables
        df = df.dropna(subset=self.target_columns)
        
        print(f"📊 Final dataset: {len(df)} samples, {len(self.feature_columns)} features")
        print(f"🎯 Base features: {len(available_base_features)}")
        print(f"🌐 Meteomatics features: {len(available_meteomatics_features)}")
        print(f"📉 Removed {initial_size - len(df)} rows with missing data")
        
        return df

    def train_models(self, df):
        """Train enhanced models with cross-validation"""
        print(f"\n🚀 Training enhanced models for {len(self.target_columns)} targets...")
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Split data temporally
        train_size = int(0.8 * len(df))
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        
        # Scale features with RobustScaler (better for outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        results = {}
        
        for target in self.target_columns:
            print(f"\n🎯 Training {self.target_names[target]} model...")
            
            # Prepare target
            y = df[target]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            
            # Conservative model parameters to prevent overfitting
            if target == 'RR':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=10,
                    min_samples_leaf=8,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            elif target == 'Tavg':
                model = GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.06,
                    max_depth=4,
                    min_samples_split=8,
                    min_samples_leaf=6,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=8,
                    min_samples_leaf=6,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            
            # Cross-validation with TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=tscv, scoring='r2', n_jobs=N_JOBS
            )
            
            # Train final model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Apply constraints
            if target == 'RR':
                y_pred = np.maximum(y_pred, 0)
            elif target == 'ss':
                y_pred = np.clip(y_pred, 0, 24)
            elif target == 'ff_avg':
                y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
            
            # Store results
            self.models[target] = model
            self.cv_results[target] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            results[target] = {
                'model': model,
                'metrics': metrics,
                'cv_results': self.cv_results[target],
                'y_test': y_test,
                'y_pred': y_pred,
                'test_dates': df.iloc[train_size:]['Tanggal']
            }
            
            print(f"  📈 R² Score: {metrics['R2']:.4f}")
            print(f"  📊 CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  📏 RMSE: {metrics['RMSE']:.4f}")
        
        return results

    def plot_predictions(self, results):
        """Create prediction plots"""
        print("\n📊 Creating prediction plots...")
        
        n_targets = len(results)
        fig, axes = plt.subplots(n_targets, 1, figsize=(15, 4*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i, (target, data) in enumerate(results.items()):
            ax = axes[i]
            
            dates = data['test_dates']
            y_test = data['y_test']
            y_pred = data['y_pred']
            metrics = data['metrics']
            cv_results = data['cv_results']
            
            ax.plot(dates, y_test, 'o-', label='Actual', alpha=0.7, markersize=3)
            ax.plot(dates, y_pred, 'x-', label='Predicted', alpha=0.7, markersize=3)
            
            ax.set_title(f'{self.target_names[target]} - R²: {metrics["R2"]:.3f} | CV: {cv_results["cv_mean"]:.3f}±{cv_results["cv_std"]:.3f}')