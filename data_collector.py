import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from src.config import RAW_DATA_DIR, WEATHER_API_URL
import os

class WeatherDataCollector:
    def __init__(self, latitude=28.6139, longitude=77.2090):  # Default: New Delhi
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = WEATHER_API_URL
    
    def collect_historical_data(self, start_date, end_date):
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,cloud_cover',
            'timezone': 'auto'
        }
        
        print(f"Collecting weather data from {start_date} to {end_date}...")
        response = requests.get(self.base_url, params=params)
        
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
            raise Exception(f"API request failed: {response.status_code}")
    
    def collect_and_save(self, years=2):
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=365*years)
        
        df = self.collect_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        filepath = os.path.join(RAW_DATA_DIR, 'weather_data.csv')
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        print(f"Total records: {len(df)}")
        return df

if __name__ == "__main__":
    collector = WeatherDataCollector()
    collector.collect_and_save(years=2)