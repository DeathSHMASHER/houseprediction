# 🏡 Real Estate Valuation Engine: House Price Prediction
### *Capstone Project 2: End-to-End Machine Learning Regression Pipeline & Interactive Deployment*

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary & Project Context

This repository contains the complete implementation of **Project 2: House Price Prediction** from the **Data Science Capstone Project Curriculum**. 

The primary objective is to engineer an enterprise-grade regression pipeline that predicts fair market values for residential real estate properties based on physical attributes (square footage, bedrooms, bathrooms, age) and neighborhood locations.

The project demonstrates:
- Data pipeline architecture and realistic data synthesis
- Exploratory Data Analysis (EDA) and publication-quality visualization
- Statistical data cleaning (imputation of missing values and IQR outlier mitigation)
- Feature engineering (One-Hot categorical encoding and StandardScaler normalization)
- Multi-model benchmarking across baseline, regularized ($L_1$, $L_2$), and ensemble tree-based algorithms
- Real-time model inference and an interactive web dashboard built with **Streamlit**

---

## 📁 Repository Directory Structure

```text
House_Price_Prediction/
│
├── data/
│   └── housing.csv                     # Ingested dataset with missing values & realistic outliers
├── notebooks/
│   └── house_price_prediction.ipynb    # Complete step-by-step reproducible Jupyter Notebook
├── src/
│   ├── generate_data.py                # Synthetic realistic dataset generator
│   ├── eda.py                          # Exploratory Data Analysis & visual plotting
│   ├── preprocess.py                   # Imputation (Median/Mode), IQR Outlier filter, OneHot, Scaling
│   ├── train.py                        # Model training, comparison, and evaluation pipeline
│   ├── predict.py                      # Standalone CLI prediction module
│   └── build_notebook.py               # Notebook synchronization generator
├── models/
│   ├── house_price_model.pkl           # Champion model pipeline (XGBoost + Preprocessor)
│   └── model_comparison.json          # Benchmark evaluation metrics (R², RMSE, MAE)
├── plots/
│   ├── price_distribution.png          # Target distribution plot (KDE + Histogram)
│   ├── area_vs_price.png               # Area vs price scatter plot across locations
│   ├── bedrooms_bathrooms_vs_price.png # Multi-panel room count boxplots
│   └── correlation_heatmap.png         # Numeric feature correlation matrix
├── app.py                              # Interactive Streamlit Web Dashboard (Bonus Deployment)
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Clean repository rules
└── README.md                           # Comprehensive documentation & methodology
```

---

## 🔬 What Was Done and Why: Comprehensive Methodology

### 1. Dataset Selection & Schema Design (`data/housing.csv`, `src/generate_data.py`)
* **What was done:** We generated a high-fidelity 2,500-sample tabular dataset containing:
  - `Area`: Continuous living area in square feet ($550 - 5,200\text{ sq ft}$)
  - `Bedrooms`: Discrete room count ($1 - 6$)
  - `Bathrooms`: Discrete bathroom count ($1 - 5$)
  - `Location`: Neighborhood category (`Downtown`, `Suburbs`, `Uptown`, `Waterfront`, `Rural`)
  - `Age`: Construction age in years ($0 - 55$)
  - `Price`: Target market valuation ($100,000 - $2,200,000$)
* **Why this was done:**
  - The Capstone specification listed **California Housing** or **Ames Housing**.
  - **California Housing** (`sklearn.datasets.fetch_california_housing`) only contains aggregate census tract block averages (`MedInc`, `AveRooms`, `AveBedrms`, `Latitude`, `Longitude`). It does **not** have individual house bedroom/bathroom counts or named neighborhood zones, and contains **zero missing values** (preventing students from executing Step 3).
  - Raw **Ames Housing** contains 80 features and requires Kaggle API authentication.
  - To fulfill **every single criterion in the curriculum** (including Step 2 EDA of Area/Beds/Baths, Step 3 Median/Mode imputation, Step 4 IQR outlier handling, Step 5 One-Hot location encoding, and Step 9 manual inputs for `Area, Bedrooms, Bathrooms, Location`), we synthesized a calibrated dataset mirroring Ames market dynamics with deliberate missing values (~2.5%) and realistic outliers.

---

### 2. Exploratory Data Analysis (EDA) (`src/eda.py`)
* **What was done:** Generated 4 visualizations saved to `plots/`:
  1. `price_distribution.png`: Evaluates target variable skewness with a normal KDE overlay.
  2. `area_vs_price.png`: Scatter plot confirming strong positive linear-to-polynomial correlation between living area and sale price, stratified by neighborhood.
  3. `bedrooms_bathrooms_vs_price.png`: Boxplots illustrating incremental valuation premiums for additional bedrooms and bathrooms.
  4. `correlation_heatmap.png`: Pearson correlation matrix highlighting inter-feature collinearity.
* **Why this was done:** EDA is essential to confirm regression assumptions, identify non-linear relationships, detect skewness, and assess feature importance before feeding data into estimators.

---

### 3. Missing Value Handling (`src/preprocess.py`)
* **What was done:**
  - Continuous numerical features (`Area`, `Bathrooms`, `Age`) are imputed using **Median** (`SimpleImputer(strategy='median')`).
  - Categorical features (`Location`) are imputed using **Mode** (`SimpleImputer(strategy='most_frequent')`).
* **Why this was done:**
  - Real-world housing data frequently contains missing entries.
  - **Median** imputation is mathematically superior to mean imputation because real estate figures are skewed and vulnerable to extreme values.
  - **Mode** (most frequent occurrence) preserves discrete category labels without introducing invalid synthetic categories.

---

### 4. Outlier Detection & Mitigation (`src/preprocess.py`)
* **What was done:** Implemented the **Interquartile Range (IQR) method**:
  $$IQR = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q_3 + 1.5 \times IQR$$
  Outlying records outside this boundary (39 records detected) were pruned prior to training.
* **Why this was done:** Ordinary Least Squares (OLS) and gradient boosting algorithms optimize squared residual loss ($\sum (y_i - \hat{y}_i)^2$). Extreme anomalies disproportionately pull the regression hyperplanes, degrading generalization on normal houses.

---

### 5. Feature Encoding & Normalization (`src/preprocess.py`)
* **What was done:**
  - **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`): Applied to `Location`.
  - **StandardScaler**: Applied to all continuous numeric features ($z = \frac{x - \mu}{\sigma}$).
  - Unified into a scikit-learn `ColumnTransformer`.
* **Why this was done:**
  - Categorical locations have no natural ordinal sequence (`Waterfront` is not mathematically "greater than" `Rural`). One-hot encoding creates orthogonal binary indicator features.
  - Gradient descent and regularized algorithms ($L_1$ / $L_2$) require features on identical scales; otherwise, coefficients with larger absolute magnitudes are penalized unfairly.

---

### 6. Train-Test Split & Modeling (`src/train.py`)
* **What was done:** Split data 80% for training (1,968 samples) and 20% for testing (493 samples) with `random_state=42`. Trained 5 candidate algorithms:
  1. **Linear Regression:** Standard baseline OLS model.
  2. **Ridge Regression ($L_2$):** Penalizes sum of squared coefficients ($\alpha \sum w_j^2$) to mitigate multicollinearity.
  3. **Lasso Regression ($L_1$):** Penalizes sum of absolute coefficients ($\alpha \sum |w_j|$) to perform automatic feature shrinkage.
  4. **Random Forest Regressor:** Bagging ensemble of 100 de-correlated decision trees.
  5. **XGBoost Regressor:** Extreme Gradient Boosting iteratively correcting pseudo-residuals.
* **Why this was done:** Directly satisfies Step 7 of the Capstone guide to compare baseline linear methods against regularized formulations and non-linear tree ensembles.

---

### 7. Model Evaluation & Benchmark Results (`src/train.py`)

All models were evaluated on the 20% unseen test partition across three evaluation metrics:
* **$R^2$ Score (Coefficient of Determination):** Proportion of variance in price explained by the features (closer to 1.0 is better).
* **RMSE (Root Mean Squared Error):** Standard deviation of the prediction residuals in dollar terms.
* **MAE (Mean Absolute Error):** Average magnitude of errors in dollar terms.

#### 📊 Benchmark Comparison Table

| Model Rank | Algorithm | $R^2$ Score | RMSE ($) | MAE ($) | Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 **1** | **XGBoost Regressor** | **0.9695** | **$39,775.24** | **$28,858.58** | **Selected Champion Model** |
| 🥈 **2** | **Random Forest Regressor** | **0.9639** | **$43,290.49** | **$31,310.02** | Ensemble Bagging |
| 🥉 **3** | **Linear Regression** | **0.9489** | **$51,509.51** | **$37,762.37** | Baseline OLS |
| 4 | **Lasso Regression ($L_1$)** | **0.9489** | **$51,521.88** | **$37,778.92** | Sparsity Regularization |
| 5 | **Ridge Regression ($L_2$)** | **0.9489** | **$51,529.11** | **$37,787.90** | Weight Shrinkage |

*Conclusion:* **XGBoost** achieved the highest $R^2$ (0.9695) and lowest prediction error ($28,858 MAE), successfully capturing non-linear interactions between square footage and neighborhood multiplier curves.

---

### 8. Serialization & Real-Time Inference (`src/predict.py`)
* **What was done:** Serialized the complete fitted pipeline (`ColumnTransformer` + `XGBoost`) into `models/house_price_model.pkl`. Created `src/predict.py` allowing instant programmatic or CLI predictions.
* **Step 9 Sample Test:**
  - **Inputs:** Area = `2,400 sq ft`, Bedrooms = `4`, Bathrooms = `3`, Location = `Waterfront`
  - **Estimated Price:** **`$1,111,896.12`**

---

### 9. Interactive Web Dashboard (`app.py` - Streamlit Bonus)
* **What was done:** Deployed a responsive web application powered by **Streamlit** containing:
  - **Tab 1: Real-time Property Predictor:** Interactive sliders for Area ($600 - 5000$), Bedrooms ($1-6$), Bathrooms ($1-5$), Age ($0-50$), and a Location dropdown. Renders the estimated price and calculated price-per-square-foot.
  - **Tab 2: Model Benchmarking:** Interactive data table and bar chart comparing $R^2$ and RMSE across all 5 models.
  - **Tab 3: EDA Gallery:** High-resolution visualizations of all distribution, room count, and correlation charts.
  - **Tab 4: Architecture & Methodology:** In-depth technical breakdown of each data science stage.

---

## 🚀 How to Run the Project Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Run EDA
```bash
py -3 src/generate_data.py
py -3 src/eda.py
```

### 4. Train Models & Benchmark
```bash
py -3 src/train.py
```

### 5. Run Single Property Prediction
```bash
py -3 src/predict.py
```

### 6. Launch the Streamlit Interactive App
```bash
py -3 -m streamlit run app.py
```
Open your browser and navigate to:
👉 `http://localhost:8501`

---

## 🎓 Skills & Competencies Demonstrated
* **Predictive Modeling:** Multi-variable regression, gradient boosting, tree ensembles.
* **Data Wrangling:** Imputation algorithms, outlier detection with IQR, category encoding.
* **Pipeline Engineering:** Scikit-Learn `Pipeline` and `ColumnTransformer` for reproducible deployments without data leakage.
* **Full-Stack ML Deployment:** Real-time web deployment with Streamlit and model serialization with `joblib`.
