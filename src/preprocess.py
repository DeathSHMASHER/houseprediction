import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def remove_outliers_iqr(df, target_col='Price', factor=1.5):
    """
    Step 4: Outlier Detection and Removal via Interquartile Range (IQR) method.
    """
    df_cleaned = df.copy()
    q1 = df_cleaned[target_col].quantile(0.25)
    q3 = df_cleaned[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    
    outliers_mask = (df_cleaned[target_col] < lower_bound) | (df_cleaned[target_col] > upper_bound)
    n_outliers = outliers_mask.sum()
    print(f"[Outlier Detection] Found {n_outliers} outliers in '{target_col}' outside [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    
    df_cleaned = df_cleaned[~outliers_mask].reset_index(drop=True)
    return df_cleaned

def get_preprocessor_pipeline(num_cols, cat_cols):
    """
    Builds the scikit-learn preprocessing pipeline:
    - Step 3: Imputation (Median for numerical, Mode for categorical)
    - Step 5: Encoding (One-Hot Encoding for categorical)
    - Step 6: Scaling (StandardScaler for numerical)
    """
    # Numerical pipeline: Median Imputer -> StandardScaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Mode Imputer -> OneHotEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    return preprocessor

if __name__ == '__main__':
    df = pd.read_csv("data/housing.csv")
    print(f"Original shape: {df.shape}")
    df_clean = remove_outliers_iqr(df, target_col='Price')
    print(f"Cleaned shape after IQR removal: {df_clean.shape}")
    
    num_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    cat_features = ['Location']
    preprocessor = get_preprocessor_pipeline(num_features, cat_features)
    
    X = df_clean.drop(columns=['Price'])
    X_trans = preprocessor.fit_transform(X)
    print(f"Transformed feature matrix shape: {X_trans.shape}")
