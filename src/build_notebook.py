import json
import os

def build_notebook(output_path="notebooks/house_price_prediction.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Capstone Project 2: House Price Prediction\n",
                "## Objective\n",
                "Predict estimated house prices based on residential property attributes using end-to-end regression machine learning pipelines."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Libraries and Load Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "df = pd.read_csv('../data/housing.csv')\n",
                "print('Dataset Shape:', df.shape)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Exploratory Data Analysis (EDA)\n",
                "- Summary statistics\n",
                "- Price distribution\n",
                "- Area vs Price\n",
                "- Bedrooms and Bathrooms vs Price\n",
                "- Correlation heatmap"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(df.info())\n",
                "print(df.describe())\n",
                "\n",
                "# 1. Price Distribution\n",
                "plt.figure(figsize=(8, 4))\n",
                "sns.histplot(df['Price'].dropna(), kde=True, color='royalblue', bins=35)\n",
                "plt.title('House Price Distribution')\n",
                "plt.xlabel('Price ($)')\n",
                "plt.show()\n",
                "\n",
                "# 2. Area vs Price\n",
                "plt.figure(figsize=(8, 4))\n",
                "sns.scatterplot(data=df, x='Area', y='Price', hue='Location', alpha=0.7)\n",
                "plt.title('Area vs Price by Location')\n",
                "plt.show()\n",
                "\n",
                "# 3. Room Counts vs Price\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                "sns.boxplot(data=df, x='Bedrooms', y='Price', ax=axes[0])\n",
                "axes[0].set_title('Bedrooms vs Price')\n",
                "sns.boxplot(data=df, x='Bathrooms', y='Price', ax=axes[1])\n",
                "axes[1].set_title('Bathrooms vs Price')\n",
                "plt.show()\n",
                "\n",
                "# 4. Correlation Heatmap\n",
                "plt.figure(figsize=(7, 5))\n",
                "sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')\n",
                "plt.title('Correlation Heatmap')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Handle Missing Values\n",
                "- Numerical columns: Impute using **Median**\n",
                "- Categorical columns: Impute using **Mode**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Missing Values Before Imputation:')\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Numerical columns impute with median\n",
                "for col in ['Area', 'Bathrooms']:\n",
                "    df[col] = df[col].fillna(df[col].median())\n",
                "\n",
                "# Categorical column impute with mode\n",
                "df['Location'] = df['Location'].fillna(df['Location'].mode()[0])\n",
                "\n",
                "print('\\nMissing Values After Imputation:')\n",
                "print(df.isnull().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Outlier Detection using IQR Method"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "q1 = df['Price'].quantile(0.25)\n",
                "q3 = df['Price'].quantile(0.75)\n",
                "iqr = q3 - q1\n",
                "lower_bound = q1 - 1.5 * iqr\n",
                "upper_bound = q3 + 1.5 * iqr\n",
                "\n",
                "outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]\n",
                "print(f'Detected {len(outliers)} outliers outside [{lower_bound:,.0f}, {upper_bound:,.0f}]')\n",
                "\n",
                "# Filter outliers\n",
                "df_cleaned = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)].reset_index(drop=True)\n",
                "print('Dataset shape after outlier removal:', df_cleaned.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5 & 6: Feature Engineering, Encoding & Scaling\n",
                "- One-Hot Encoding for categorical features (`Location`)\n",
                "- StandardScaler for numeric features (`Area`, `Bedrooms`, `Bathrooms`, `Age`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "X = df_cleaned.drop(columns=['Price'])\n",
                "y = df_cleaned['Price']\n",
                "\n",
                "num_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']\n",
                "cat_features = ['Location']\n",
                "\n",
                "# Train-Test Split (Step 6)\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "print(f'Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7 & 8: Build Models and Evaluate\n",
                "- Linear Regression\n",
                "- Ridge Regression\n",
                "- Lasso Regression\n",
                "- Random Forest Regressor\n",
                "- XGBoost Regressor\n",
                "\n",
                "Metrics: RMSE, MAE, R² Score"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from xgboost import XGBRegressor\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', StandardScaler(), num_features),\n",
                "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)\n",
                "    ]\n",
                ")\n",
                "\n",
                "models = {\n",
                "    'Linear Regression': LinearRegression(),\n",
                "    'Ridge Regression': Ridge(alpha=1.0),\n",
                "    'Lasso Regression': Lasso(alpha=50.0),\n",
                "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),\n",
                "    'XGBoost': XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)\n",
                "}\n",
                "\n",
                "results = []\n",
                "for name, model in models.items():\n",
                "    pipe = Pipeline([('prep', preprocessor), ('reg', model)])\n",
                "    pipe.fit(X_train, y_train)\n",
                "    preds = pipe.predict(X_test)\n",
                "    \n",
                "    r2 = r2_score(y_test, preds)\n",
                "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
                "    mae = mean_absolute_error(y_test, preds)\n",
                "    results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})\n",
                "\n",
                "results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)\n",
                "display(results_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 9: Prediction & Model Serialization\n",
                "Input: Area, Bedrooms, Bathrooms, Location -> Output: Estimated House Price"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import joblib\n",
                "\n",
                "# Best model pipeline\n",
                "best_pipe = Pipeline([('prep', preprocessor), ('reg', models['XGBoost'])])\n",
                "best_pipe.fit(X_train, y_train)\n",
                "joblib.dump(best_pipe, '../models/house_price_model.pkl')\n",
                "print('Model saved successfully to ../models/house_price_model.pkl')\n",
                "\n",
                "# Sample Inference\n",
                "sample_house = pd.DataFrame([{\n",
                "    'Area': 2500,\n",
                "    'Bedrooms': 4,\n",
                "    'Bathrooms': 3,\n",
                "    'Location': 'Waterfront',\n",
                "    'Age': 5\n",
                "}])\n",
                "pred_price = best_pipe.predict(sample_house)[0]\n",
                "print(f'Predicted Price for Sample House: ${pred_price:,.2f}')"
            ]
        }
    ]
    
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"Jupyter Notebook successfully built at {output_path}")

if __name__ == '__main__':
    build_notebook()
