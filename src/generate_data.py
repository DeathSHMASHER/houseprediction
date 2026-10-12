import numpy as np
import pandas as pd
import os

def generate_housing_data(n_samples=2500, random_seed=42):
    np.random.seed(random_seed)
    
    locations = ['Downtown', 'Suburbs', 'Uptown', 'Waterfront', 'Rural']
    location_weights = [0.25, 0.35, 0.20, 0.08, 0.12]
    location_multipliers = {
        'Downtown': 1.45,
        'Suburbs': 1.05,
        'Uptown': 1.30,
        'Waterfront': 1.85,
        'Rural': 0.75
    }
    
    # 1. Area (sq ft) - lognormal distribution to simulate realistic house sizes
    area = np.random.normal(loc=1950, scale=650, size=n_samples)
    area = np.clip(area, 550, 5200).round()
    
    # 2. Bedrooms correlated with area
    bedrooms = np.clip(np.round(area / 600 + np.random.normal(0, 0.6, n_samples)), 1, 6).astype(int)
    
    # 3. Bathrooms correlated with bedrooms
    bathrooms = np.clip(np.round(bedrooms * 0.65 + np.random.normal(0, 0.4, n_samples) * 0.5), 1, 5).astype(int)
    
    # 4. Age of the house (years)
    age = np.random.randint(0, 55, size=n_samples)
    
    # 5. Location
    loc_choices = np.random.choice(locations, size=n_samples, p=location_weights)
    
    # Base price calculation with realistic market economics:
    # Base: $75,000 + $135/sqft + $22,000/bed + $32,000/bath - $1,100/year_age + location multiplier + noise
    base_price = (
        75000 +
        area * 140 +
        bedrooms * 24000 +
        bathrooms * 35000 -
        age * 950
    )
    
    mult_array = np.array([location_multipliers[l] for l in loc_choices])
    price = base_price * mult_array + np.random.normal(0, 28000, size=n_samples)
    price = np.round(np.clip(price, 85000, 2200000), -2)
    
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location': loc_choices,
        'Age': age,
        'Price': price
    })
    
    # Introduce deliberate missing values (Step 3: Handle Missing Values with Median/Mode)
    # ~3% missing in Area, Bathrooms, and Location
    nan_area_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
    nan_bath_idx = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    nan_loc_idx = np.random.choice(df.index, size=int(0.025 * n_samples), replace=False)
    
    df.loc[nan_area_idx, 'Area'] = np.nan
    df.loc[nan_bath_idx, 'Bathrooms'] = np.nan
    df.loc[nan_loc_idx, 'Location'] = np.nan
    
    # Introduce deliberate outliers for Step 4 (IQR Outlier Detection demonstration)
    outlier_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_idx, 'Price'] = df.loc[outlier_idx, 'Price'] * np.random.uniform(2.2, 3.0, size=15)
    
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'housing.csv')
    df = generate_housing_data(n_samples=2500)
    df.to_csv(csv_path, index=False)
    print(f"Generated housing dataset with {len(df)} records at {csv_path}")
    print(df.info())
