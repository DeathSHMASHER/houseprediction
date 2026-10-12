import os
import joblib
import pandas as pd

def predict_house_price(area, bedrooms, bathrooms, location, age=10, model_path="models/house_price_model.pkl"):
    """
    Step 9: Takes Area, Bedrooms, Bathrooms, Location (and optional Age)
    and predicts the estimated house price.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    model_pipeline = joblib.load(model_path)
    
    input_df = pd.DataFrame([{
        'Area': float(area),
        'Bedrooms': int(bedrooms),
        'Bathrooms': float(bathrooms),
        'Location': str(location),
        'Age': int(age)
    }])
    
    predicted_price = model_pipeline.predict(input_df)[0]
    return float(predicted_price)

if __name__ == '__main__':
    # Example Step 9 prediction
    sample_area = 2400
    sample_beds = 4
    sample_baths = 3
    sample_location = 'Waterfront'
    
    price = predict_house_price(sample_area, sample_beds, sample_baths, sample_location)
    print("=== Step 9: Prediction Sample ===")
    print(f"Inputs: Area={sample_area} sq ft, Bedrooms={sample_beds}, Bathrooms={sample_baths}, Location='{sample_location}'")
    print(f"Predicted House Price: ${price:,.2f}")
