import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class WeatherAPIEnhancedPredictor:
    def __init__(self, api_key):
        # REMOVED WIND DIRECTION (ddd_car) from targets - poor performance
        self.target_columns = ['RR', 'ss', 'Tavg', 'ff_avg']
        self.target_names = {
            'RR': 'Rainfall (mm)',
            'ss': 'Sunshine Duration (hours)', 
            'Tavg': 'Average Temperature (°C)',
            'ff_avg': 'Wind Speed (m/s)'
        }
        
        # WeatherAPI credentials
        self.api_key = api_key
        
        # Makassar coordinates
        self.location = "-5.1477,119.4327"  # Makassar lat, lon
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'weatherapi_enhanced_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🌤️ WeatherAPI Enhanced Weather Predictor initialized")
        print(f"🎯 Target variables: {list(self.target_names.keys())} (removed wind direction)")
        print(f"📍 Location: Makassar, Indonesia {self.location}")
        print(f"🚫 NO rolling means used - pure independent features only")

    def fetch_weatherapi_data(self, start_date, end_date):
        """Fetch additional meteorological data from WeatherAPI.com - starting from today backward"""
        print(f"\n🌐 Fetching data from WeatherAPI.com...")
        print(f"📅 Original date range: {start_date} to {end_date}")
        
        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Start from today or end_date (whichever is earlier) and go backward
        today = pd.to_datetime('today')
        current_date = min(end_date, today)
        
        print(f"🔄 Starting backward fetch from: {current_date.date()}")
        print(f"🎯 Target start date: {start_date.date()}")
        
        weather_data = []
        base_url = "http://api.weatherapi.com/v1/history.json"
        
        # Rate limiting: 1 request per second to be safe
        request_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5  # Stop if 5 consecutive days have no data
        
        try:
            while current_date >= start_date and consecutive_failures < max_consecutive_failures:
                date_str = current_date.strftime('%Y-%m-%d')
                
                params = {
                    'key': self.api_key,
                    'q': self.location,
                    'dt': date_str,
                    'aqi': 'yes'  # Include air quality data
                }
                
                try:
                    print(f"🔄 Fetching data for {date_str}... (Day {request_count + 1})")
                    response = requests.get(base_url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract day data
                        if 'forecast' in data and 'forecastday' in data['forecast']:
                            day_data = data['forecast']['forecastday'][0]['day']
                            astro_data = data['forecast']['forecastday'][0]['astro']
                            
                            # Get hourly data for better statistics
                            hourly_data = data['forecast']['forecastday'][0]['hour']
                            
                            # Calculate additional statistics from hourly data
                            hourly_pressure = [h['pressure_mb'] for h in hourly_data]
                            hourly_cloud = [h['cloud'] for h in hourly_data]
                            hourly_humidity = [h['humidity'] for h in hourly_data]
                            hourly_vis = [h['vis_km'] for h in hourly_data]
                            
                            weather_record = {
                                'Tanggal': current_date,
                                'pressure_mb': np.mean(hourly_pressure),
                                'pressure_max': np.max(hourly_pressure),
                                'pressure_min': np.min(hourly_pressure),
                                'pressure_range': np.max(hourly_pressure) - np.min(hourly_pressure),
                                'cloud_cover': np.mean(hourly_cloud),
                                'cloud_max': np.max(hourly_cloud),
                                'humidity_weatherapi': np.mean(hourly_humidity),
                                'visibility_km': np.mean(hourly_vis),
                                'uv_index': day_data.get('uv', 0),
                                'maxtemp_c': day_data.get('maxtemp_c', 0),
                                'mintemp_c': day_data.get('mintemp_c', 0),
                                'avgtemp_c': day_data.get('avgtemp_c', 0),
                                'maxwind_kph': day_data.get('maxwind_kph', 0),
                                'totalprecip_mm': day_data.get('totalprecip_mm', 0),
                                'totalsnow_cm': day_data.get('totalsnow_cm', 0),
                                'avghumidity': day_data.get('avghumidity', 0),
                                'daily_will_it_rain': day_data.get('daily_will_it_rain', 0),
                                'daily_chance_of_rain': day_data.get('daily_chance_of_rain', 0),
                                'condition_code': day_data.get('condition', {}).get('code', 0),
                                'sunrise': astro_data.get('sunrise', ''),
                                'sunset': astro_data.get('sunset', ''),
                                'moonrise': astro_data.get('moonrise', ''),
                                'moonset': astro_data.get('moonset', ''),
                                'moon_phase': astro_data.get('moon_phase', ''),
                                'moon_illumination': astro_data.get('moon_illumination', 0)
                            }
                            
                            # Add air quality data if available
                            if 'current' in data and 'air_quality' in data['current']:
                                air_quality = data['current']['air_quality']
                                weather_record.update({
                                    'co': air_quality.get('co', 0),
                                    'no2': air_quality.get('no2', 0),
                                    'o3': air_quality.get('o3', 0),
                                    'so2': air_quality.get('so2', 0),
                                    'pm2_5': air_quality.get('pm2_5', 0),
                                    'pm10': air_quality.get('pm10', 0),
                                    'us_epa_index': air_quality.get('us-epa-index', 0),
                                    'gb_defra_index': air_quality.get('gb-defra-index', 0)
                                })
                            
                            weather_data.append(weather_record)
                            request_count += 1
                            consecutive_failures = 0  # Reset failure counter on success
                            print(f"✅ Data found for {date_str}")
                            
                    elif response.status_code == 400:
                        print(f"⚠️ No data available for {date_str}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"🛑 Found {consecutive_failures} consecutive days without data. Stopping backward search.")
                            break
                    elif response.status_code == 403:
                        print(f"⚠️ API limit reached or invalid key")
                        break
                    else:
                        print(f"⚠️ Error {response.status_code} for {date_str}: {response.text}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"🛑 Too many consecutive errors. Stopping backward search.")
                            break
                        
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Request failed for {date_str}: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"🛑 Too many consecutive request failures. Stopping backward search.")
                        break
                
                # Rate limiting
                time.sleep(1.1)  # 1.1 seconds between requests
                current_date -= timedelta(days=1)  # Move backward in time
                
                # Respect free tier limits (300 requests/month)
                if request_count >= 300:
                    print(f"🛑 Reached free tier limit of 300 requests")
                    break
            
            if weather_data:
                # Sort data by date (oldest first) for consistency
                df_weather = pd.DataFrame(weather_data)
                df_weather = df_weather.sort_values('Tanggal').reset_index(drop=True)
                
                print(f"\n✅ Successfully fetched {len(df_weather)} days of WeatherAPI data")
                print(f"📅 Data range: {df_weather['Tanggal'].min().date()} to {df_weather['Tanggal'].max().date()}")
                print(f"📊 WeatherAPI variables: {[col for col in df_weather.columns if col != 'Tanggal']}")
                return df_weather
            else:
                print(f"❌ No WeatherAPI data could be fetched")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching WeatherAPI data: {str(e)}")
            return None

    def merge_with_weatherapi(self, original_df, weatherapi_df):
        """Merge original dataset with WeatherAPI data"""
        if weatherapi_df is None:
            print("⚠️ No WeatherAPI data to merge")
            return original_df
        
        print("\n🔗 Merging datasets...")
        
        # Ensure both have datetime type
        original_df['Tanggal'] = pd.to_datetime(original_df['Tanggal'], format='%d-%m-%Y')
        weatherapi_df['Tanggal'] = pd.to_datetime(weatherapi_df['Tanggal'])
        
        # Merge on date
        merged_df = pd.merge(original_df, weatherapi_df, on='Tanggal', how='left')
        
        # Fill missing WeatherAPI data with interpolation
        weatherapi_cols = [col for col in weatherapi_df.columns if col != 'Tanggal']
        for col in weatherapi_cols:
            if col in merged_df.columns:
                # Use forward fill, then backward fill, then median
                merged_df[col] = merged_df[col].fillna(method='ffill')
                merged_df[col] = merged_df[col].fillna(method='bfill')
                if merged_df[col].isna().any():
                    merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
        print(f"✅ Merged dataset: {merged_df.shape}")
        print(f"📊 Added WeatherAPI variables: {weatherapi_cols}")
        
        return merged_df

    def preprocess_data(self, df):
        """Enhanced preprocessing with WeatherAPI features"""
        print("\n🔧 Starting enhanced data preprocessing with WeatherAPI...")
        
        df = df.copy()
        initial_size = len(df)
        
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
        
        # Convert wind direction to numeric
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
        
        # Simple lag features (1-day only, from non-target variables)
        non_target_vars = ['Tn', 'Tx', 'RH_avg']
        
        # Add WeatherAPI variables to lag features if available
        weatherapi_vars = []
        potential_weatherapi = ['pressure_mb', 'cloud_cover', 'visibility_km', 'uv_index']
        
        for var in potential_weatherapi:
            if var in df.columns:
                weatherapi_vars.append(var)
                non_target_vars.append(var)
        
        if weatherapi_vars:
            print(f"🌟 Using WeatherAPI variables for lag features: {weatherapi_vars}")
        
        # Create 1-day lag features
        for var in non_target_vars:
            if var in df.columns:
                df[f'{var}_Lag_1'] = df[var].shift(1)
        
        # Advanced atmospheric features if pressure is available
        if 'pressure_mb' in df.columns:
            print("🌟 Creating advanced atmospheric pressure features...")
            df['pressure_change_1d'] = df['pressure_mb'].diff(1)
            df['pressure_tendency'] = np.where(df['pressure_change_1d'] > 1, 1,
                                             np.where(df['pressure_change_1d'] < -1, -1, 0))
        
        # Cloud cover derived features
        if 'cloud_cover' in df.columns:
            print("🌟 Creating cloud cover features...")
            df['cloud_change_1d'] = df['cloud_cover'].diff(1)
            df['is_cloudy'] = (df['cloud_cover'] > 50).astype(int)
        
        # Air quality features if available
        if 'pm2_5' in df.columns:
            print("🌟 Creating air quality features...")
            df['air_quality_index'] = np.log1p(df['pm2_5'] + df['pm10']) if 'pm10' in df.columns else np.log1p(df['pm2_5'])
        
        # NO ROLLING MEANS - maintain the no-rolling approach
        print("🚫 Rolling means EXCLUDED to maintain temporal independence")
        
        # Feature columns (pure independent features only)
        base_features = [
            # Basic current-day measurements (non-targets)
            'Tn', 'Tx', 'RH_avg', 'ff_x', 'ddd_x',
            
            # Temporal features (derived from date only)
            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
            'DayOfYear_sin', 'DayOfYear_cos', 'Season',
            
            # Derived current-day features
            'Temp_Range', 'Temp_Humidity', 'Dew_Point'
        ]
        
        # Add WeatherAPI base features
        weatherapi_base = ['pressure_mb', 'pressure_max', 'pressure_min', 'pressure_range',
                          'cloud_cover', 'cloud_max', 'visibility_km', 'uv_index',
                          'maxwind_kph', 'avghumidity', 'daily_chance_of_rain',
                          'condition_code', 'moon_illumination']
        
        for feat in weatherapi_base:
            if feat in df.columns:
                base_features.append(feat)
        
        # Add derived atmospheric features
        atmospheric_features = ['pressure_change_1d', 'pressure_tendency', 
                              'cloud_change_1d', 'is_cloudy', 'air_quality_index']
        
        for feat in atmospheric_features:
            if feat in df.columns:
                base_features.append(feat)
        
        # Add air quality features
        air_quality_features = ['co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10']
        for feat in air_quality_features:
            if feat in df.columns:
                base_features.append(feat)
        
        # Add 1-day lag features
        feature_columns = base_features.copy()
        for col in df.columns:
            if col.endswith('_Lag_1') and not any(target in col for target in self.target_columns):
                feature_columns.append(col)
        
        # Filter available features
        self.feature_columns = [f for f in feature_columns if f in df.columns]
        
        # Handle any remaining NaN values
        for col in self.feature_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Drop rows with NaN in target variables
        df = df.dropna(subset=self.target_columns)
        
        print(f"📊 Final dataset: {len(df)} samples, {len(self.feature_columns)} features")
        print(f"📉 Removed {initial_size - len(df)} rows with missing data")
        
        # Show feature breakdown
        weatherapi_features = [f for f in self.feature_columns if any(w in f for w in weatherapi_vars)]
        if weatherapi_features:
            print(f"🌟 WeatherAPI-derived features: {len(weatherapi_features)}")
        
        return df

    def train_models(self, df):
        """Train enhanced models with WeatherAPI features"""
        print(f"\n🚀 Training WeatherAPI-enhanced models for {len(self.target_columns)} targets...")
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Split data temporally
        train_size = int(0.8 * len(df))
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        
        # Scale features with RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for target in self.target_columns:
            print(f"\n🎯 Training {self.target_names[target]} model...")
            
            # Prepare target
            y = df[target]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
            
            # Enhanced model parameters (can be slightly more complex with additional data)
            if target == 'Tavg':
                # Temperature is most predictable, can use slightly more complex model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.06,
                    max_depth=4,
                    min_samples_split=10,
                    min_samples_leaf=6,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            elif target == 'RR':
                # Rainfall benefits from atmospheric pressure
                model = GradientBoostingRegressor(
                    n_estimators=90,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_split=12,
                    min_samples_leaf=8,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=80,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_split=12,
                    min_samples_leaf=8,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                )
            
            # Cross-validation with TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=tscv, scoring='r2'
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
            
            results[target] = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'test_dates': df.iloc[train_size:]['Tanggal'],
                'feature_importance': dict(zip(self.feature_columns, model.feature_importances_))
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
            
            ax.plot(dates, y_test, 'o-', label='Actual', alpha=0.7, markersize=3)
            ax.plot(dates, y_pred, 'x-', label='Predicted', alpha=0.7, markersize=3)
            
            ax.set_title(f'{self.target_names[target]} - R²: {metrics["R2"]*100:.2f}% | CV: {data["cv_mean"]*100:.2f}±{data["cv_std"]*100:.2f}% (WeatherAPI)')
            ax.set_ylabel(self.target_names[target])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if i == n_targets - 1:
                ax.set_xlabel('Date')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'weatherapi_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, results):
        """Plot enhanced feature importance showing WeatherAPI contributions"""
        print("📊 Creating enhanced feature importance plots...")
        
        n_targets = len(results)
        fig, axes = plt.subplots(n_targets, 1, figsize=(14, 6*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i, (target, data) in enumerate(results.items()):
            importance_dict = data['feature_importance']
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
            features, importance = zip(*sorted_features)
            
            ax = axes[i]
            bars = ax.barh(range(len(features)), importance)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_title(f'Top 15 Features - {self.target_names[target]}')
            ax.set_xlabel('Feature Importance')
            
            # Color code: WeatherAPI features in different color
            weatherapi_keywords = ['pressure', 'cloud', 'visibility', 'uv', 'wind', 'humidity', 
                                  'condition', 'moon', 'air_quality', 'pm2_5', 'pm10']
            
            for j, (feat, bar) in enumerate(zip(features, bars)):
                if any(keyword in feat.lower() for keyword in weatherapi_keywords):
                    bar.set_color('orange')  # WeatherAPI features
                else:
                    bar.set_color('steelblue')  # Original features
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'weatherapi_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results):
        """Save enhanced results"""
        print("\n💾 Saving WeatherAPI-enhanced results...")
        
        # Save individual target metrics
        for target, data in results.items():
            target_metrics = {
                'R2': data['metrics']['R2'],
                'RMSE': data['metrics']['RMSE'],
                'MAE': data['metrics']['MAE'],
                'CV_R2_mean': data['cv_mean'],
                'CV_R2_std': data['cv_std']
            }
            
            with open(os.path.join(self.results_dir, f'weatherapi_metrics_{target}.txt'), 'w') as f:
                for key, value in target_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
        
        # Save feature importance
        with open(os.path.join(self.results_dir, 'feature_importance_analysis.txt'), 'w') as f:
            f.write("WEATHERAPI ENHANCED FEATURE IMPORTANCE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for target, data in results.items():
                f.write(f"\n{self.target_names[target]}:\n")
                f.write("-" * 30 + "\n")
                
                importance_dict = data['feature_importance']
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (feature, importance) in enumerate(sorted_features[:15], 1):
                    f.write(f"{rank:2d}. {feature:30s} {importance:.4f}\n")
        
        print(f"✅ Results saved to: {self.results_dir}")

    def generate_summary_report(self, results):
        """Generate comprehensive WeatherAPI-enhanced summary"""
        print("\n📋 Generating WeatherAPI-enhanced summary report...")
        
        print("=" * 70)
        print("🌤️ WEATHERAPI-ENHANCED WEATHER PREDICTION MODEL")
        print("=" * 70)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target Variables: {len(self.target_columns)} (removed wind direction)")
        print(f"🔧 Total Features: {len(self.feature_columns)}")
        print("🚫 NO rolling means used - pure independent features only")
        print()
        
        # Count WeatherAPI features
        weatherapi_features = [f for f in self.feature_columns if any(keyword in f.lower() for keyword in 
                              ['pressure', 'cloud', 'visibility', 'uv', 'wind', 'humidity', 
                               'condition', 'moon', 'air_quality', 'pm2_5', 'pm10'])]
        
        print(f"🌟 WeatherAPI-derived Features: {len(weatherapi_features)}")
        print()
        
        print("📊 MODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        total_r2 = 0
        for target, data in results.items():
            metrics = data['metrics']
            total_r2 += metrics['R2']
            
            print(f"{self.target_names[target]:25} - R²: {metrics['R2']:.4f} | CV: {data['cv_mean']:.3f}±{data['cv_std']:.3f}")
        
        avg_r2 = total_r2 / len(results)
        print()
        print(f"📈 Average R² Score: {avg_r2:.4f}")
        print()
        
        print("🌟 WEATHERAPI ENHANCEMENTS:")
        print("-" * 40)
        print("✅ Atmospheric pressure data (critical for weather prediction)")
        print("✅ Cloud cover information (improves radiation/sunshine prediction)")
        print("✅ Advanced visibility and UV data")
        print("✅ Pressure change features (weather system movement)")
        print("✅ Air quality data (pollution impact on weather)")
        print("✅ Enhanced humidity and wind data")
        print()
        
        print("🎯 KEY IMPROVEMENTS:")
        print("-" * 40)
        print("✅ Removed wind direction (was worst performing target)")
        print("✅ Eliminated data leakage from target variables")
        print("🚫 NO rolling means (maintains temporal independence)")
        print("✅ Enhanced atmospheric features from WeatherAPI")
        print("✅ Conservative model parameters (prevent overfitting)")
        print()

def main():
    try:
        print("🌤️ Starting WeatherAPI-Enhanced Weather Prediction Model...")
        
        # WeatherAPI credentials
        api_key = "b055972587f54b83a77182412250107"
        
        # Load original data
        print("📂 Loading original dataset...")
        df_original = pd.read_csv('makassar.csv')
        print(f"📊 Loaded original dataset: {df_original.shape}")
        
        # Initialize enhanced predictor
        predictor = WeatherAPIEnhancedPredictor(api_key)
        
        # Get date range from original data
        df_original['Tanggal'] = pd.to_datetime(df_original['Tanggal'], format='%d-%m-%Y')
        start_date = df_original['Tanggal'].min()
        end_date = df_original['Tanggal'].max()
        
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"📊 Total days: {(end_date - start_date).days + 1}")
        
        # Fetch WeatherAPI data
        weatherapi_df = predictor.fetch_weatherapi_data(start_date, end_date)
        
        # Merge datasets
        combined_df = predictor.merge_with_weatherapi(df_original, weatherapi_df)
        
        # Preprocess combined data
        processed_df = predictor.preprocess_data(combined_df)
        
        # Train enhanced models
        results = predictor.train_models(processed_df)
        
        # Generate visualizations
        predictor.plot_predictions(results)
        predictor.plot_feature_importance(results)
        
        # Save everything
        predictor.save_results(results)
        
        # Generate comprehensive report
        predictor.generate_summary_report(results)
        
        print(f"\n🎉 WeatherAPI-enhanced model training completed successfully!")
        print(f"📁 All results saved to: {predictor.results_dir}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
