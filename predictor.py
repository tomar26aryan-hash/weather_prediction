import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, WEATHER_API_URL
import os

class WeatherPredictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'))
        self.feature_cols = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'feature_cols.pkl'))
    
    def get_current_weather(self, latitude, longitude):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=2)
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,cloud_cover',
            'timezone': 'auto'
        }
        
        response = requests.get(WEATHER_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'datetime': pd.to_datetime(data['hourly']['time']),
                'temperature_2m': data['hourly']['temperature_2m'],
                'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
                'precipitation': data['hourly']['precipitation'],
                'pressure_msl': data['hourly']['pressure_msl'],
                'wind_speed_10m': data['hourly']['wind_speed_10m'],
                'cloud_cover': data['hourly']['cloud_cover']
            })
            return df
        else:
            raise Exception(f"Failed to fetch weather data: {response.status_code}")
    
    def prepare_features(self, df):
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Lagged features
        for col in ['temperature_2m', 'relative_humidity_2m', 'pressure_msl']:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_3h'] = df[col].shift(3)
            df[f'{col}_lag_6h'] = df[col].shift(6)
            df[f'{col}_lag_12h'] = df[col].shift(12)
            df[f'{col}_lag_24h'] = df[col].shift(24)
        
        # Rolling statistics
        for col in ['temperature_2m', 'wind_speed_10m', 'precipitation']:
            df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6).mean()
            df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6).std()
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
        
        df = df.dropna()
        
        # Get the latest row
        latest_data = df.iloc[-1:][self.feature_cols]
        
        return latest_data
    
    def predict(self, latitude, longitude):
        # Get current weather data
        df = self.get_current_weather(latitude, longitude)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Get current temperature for comparison
        current_temp = df['temperature_2m'].iloc[-1]
        
        return {
            'predicted_temperature': round(prediction, 2),
            'current_temperature': round(current_temp, 2),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_for': (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        }

if __name__ == "__main__":
    predictor = WeatherPredictor()
    result = predictor.predict(28.6139, 77.2090)  # New Delhi
    print(result)