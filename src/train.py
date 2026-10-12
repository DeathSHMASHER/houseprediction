import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from preprocess import remove_outliers_iqr, get_preprocessor_pipeline

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'RMSE': round(float(rmse), 2),
        'MAE': round(float(mae), 2),
        'R2': round(float(r2), 4)
    }

def train_and_evaluate(data_path="data/housing.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Handle Outliers via IQR
    df_cleaned = remove_outliers_iqr(df, target_col='Price', factor=1.5)
    
    # 3. Define Features and Target
    X = df_cleaned.drop(columns=['Price'])
    y = df_cleaned['Price']
    
    num_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    cat_features = ['Location']
    
    # 4. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 5. Define Candidate Models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=50.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        preprocessor = get_preprocessor_pipeline(num_features, cat_features)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict on Test Set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        
        print(f"[{name}] -> R²: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f}")
    
    # Find Best Model (Highest R²)
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n*** Best Model: {best_model_name} with R2: {results[best_model_name]['R2']}")
    
    # Save Best Model Pipeline
    model_path = os.path.join(models_dir, "house_price_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to {model_path}")
    
    # Save Comparison Metrics
    comparison_info = {
        'best_model': best_model_name,
        'features': {
            'numerical': num_features,
            'categorical': cat_features
        },
        'locations': sorted(df_cleaned['Location'].dropna().unique().tolist()),
        'results': results
    }
    comparison_path = os.path.join(models_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_info, f, indent=4)
    print(f"Saved comparison summary to {comparison_path}")
    
    return comparison_info

if __name__ == '__main__':
    train_and_evaluate()
