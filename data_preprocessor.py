import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import os
import joblib

class WeatherDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_raw_data(self):
        filepath = os.path.join(RAW_DATA_DIR, 'weather_data.csv')
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def create_features(self, df):
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Extract time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Create lagged features
        for col in ['temperature_2m', 'relative_humidity_2m', 'pressure_msl']:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_3h'] = df[col].shift(3)
            df[f'{col}_lag_6h'] = df[col].shift(6)
            df[f'{col}_lag_12h'] = df[col].shift(12)
            df[f'{col}_lag_24h'] = df[col].shift(24)
        
        # Create rolling statistics
        for col in ['temperature_2m', 'wind_speed_10m', 'precipitation']:
            df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6).mean()
            df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6).std()
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
        
        # Create target variable (temperature 24 hours ahead)
        df['temperature_2m_next_day'] = df['temperature_2m'].shift(-24)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self, df, test_size=0.2):
        # Select features
        feature_cols = [col for col in df.columns if col not in ['datetime', 'temperature_2m_next_day']]
        X = df[feature_cols]
        y = df['temperature_2m_next_day']
        
        # Split data (time-series split - last 20% for testing)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_cols
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, feature_cols):
        # Save data
        pd.DataFrame(X_train).to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
        pd.DataFrame(y_train, columns=['target']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
        pd.DataFrame(y_test, columns=['target']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
        
        # Save scaler and feature columns
        joblib.dump(self.scaler, os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'))
        joblib.dump(feature_cols, os.path.join(PROCESSED_DATA_DIR, 'feature_cols.pkl'))
        
        print("Processed data saved successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {len(feature_cols)}")

if __name__ == "__main__":
    preprocessor = WeatherDataPreprocessor()
    df = preprocessor.load_raw_data()
    df = preprocessor.create_features(df)
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_training_data(df)
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, feature_cols)