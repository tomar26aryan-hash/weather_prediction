#!/usr/bin/env python3

from src.data_collector import WeatherDataCollector
from src.data_preprocessor import WeatherDataPreprocessor
from src.model_trainer import WeatherModelTrainer

def main():
    print("="*60)
    print("WEATHER PREDICTION ML PROJECT - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Collect Data
    print("\n[STEP 1] Collecting Weather Data...")
    collector = WeatherDataCollector(latitude=28.6139, longitude=77.2090)
    collector.collect_and_save(years=2)
    
    # Step 2: Preprocess Data
    print("\n[STEP 2] Preprocessing Data...")
    preprocessor = WeatherDataPreprocessor()
    df = preprocessor.load_raw_data()
    df = preprocessor.create_features(df)
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_training_data(df)
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, feature_cols)
    
    # Step 3: Train Models
    print("\n[STEP 3] Training Models...")
    trainer = WeatherModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.select_best_model(results)
    trainer.save_best_model()
    trainer.plot_results(results, y_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the web app: python web_app/app.py")
    print("2. Open browser: http://localhost:5000")

if __name__ == "__main__":
    main()