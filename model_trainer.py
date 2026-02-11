import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
from src.config import PROCESSED_DATA_DIR, MODELS_DIR
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0)
        }
        self.best_model = None
        self.best_model_name = None
    
    def load_data(self):
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'))['target'].values
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'))['target'].values
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            results[name] = {
                'model': model,
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'y_pred_test': y_pred_test
            }
            
            print(f"{name} Results:")
            print(f"  Train RMSE: {results[name]['train_rmse']:.4f}")
            print(f"  Test RMSE: {results[name]['test_rmse']:.4f}")
            print(f"  Test MAE: {results[name]['test_mae']:.4f}")
            print(f"  Test R²: {results[name]['test_r2']:.4f}")
        
        return results
    
    def select_best_model(self, results):
        best_rmse = float('inf')
        
        for name, result in results.items():
            if result['test_rmse'] < best_rmse:
                best_rmse = result['test_rmse']
                self.best_model = result['model']
                self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Test RMSE: {results[self.best_model_name]['test_rmse']:.4f}")
        print(f"Test MAE: {results[self.best_model_name]['test_mae']:.4f}")
        print(f"Test R²: {results[self.best_model_name]['test_r2']:.4f}")
        print(f"{'='*50}")
    
    def save_best_model(self):
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        metadata = {
            'model_name': self.best_model_name,
            'model_path': model_path
        }
        joblib.dump(metadata, os.path.join(MODELS_DIR, 'model_metadata.pkl'))
        
        print(f"\nBest model saved to {model_path}")
    
    def plot_results(self, results, y_test):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (name, result) in enumerate(results.items()):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            ax.scatter(y_test, result['y_pred_test'], alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Temperature (°C)')
            ax.set_ylabel('Predicted Temperature (°C)')
            ax.set_title(f'{name}\nRMSE: {result["test_rmse"]:.4f}, R²: {result["test_r2"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {MODELS_DIR}/model_comparison.png")

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.select_best_model(results)
    trainer.save_best_model()
    trainer.plot_results(results, y_test)