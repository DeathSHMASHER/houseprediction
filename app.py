import os
import json
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="House Price Prediction AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 18px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
    }
    .prediction-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    model_path = os.path.join("models", "house_price_model.pkl")
    comparison_path = os.path.join("models", "model_comparison.json")
    
    if not os.path.exists(model_path) or not os.path.exists(comparison_path):
        return None, None
        
    model = joblib.load(model_path)
    with open(comparison_path, "r") as f:
        comparison = json.load(f)
        
    return model, comparison

model, comparison_data = load_model_and_metadata()

# Header
st.markdown("<div class='main-header'>🏡 Real Estate Valuation Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Capstone Project 2: End-to-End House Price Prediction Pipeline with Scikit-Learn & XGBoost</div>", unsafe_allow_html=True)

if model is None:
    st.error("Model artifacts not found! Please run `python src/train.py` first to generate the trained model.")
    st.stop()

# Tabs for structured navigation
tab_predict, tab_comparison, tab_eda, tab_pipeline = st.tabs([
    "🎯 Price Prediction",
    "📊 Model Comparison",
    "📈 Exploratory Data Analysis (EDA)",
    "📑 Pipeline & Methodology"
])

# ----------------- TAB 1: PREDICTION -----------------
with tab_predict:
    st.write("### 🏠 Input Property Attributes")
    st.caption("Adjust the property specifications below to generate an instant algorithmic price prediction.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("#### Physical Attributes")
        area = st.slider(
            "📐 Living Area (Square Feet)",
            min_value=600,
            max_value=5000,
            value=2200,
            step=50,
            help="Total heated interior floor area."
        )
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            bedrooms = st.selectbox(
                "🛏️ Bedrooms",
                options=[1, 2, 3, 4, 5, 6],
                index=2
            )
        with c_sub2:
            bathrooms = st.selectbox(
                "🚿 Bathrooms",
                options=[1, 2, 3, 4, 5],
                index=1
            )
            
        age = st.slider(
            "⏳ Property Age (Years)",
            min_value=0,
            max_value=50,
            value=8,
            help="Age of construction in years (0 indicates newly built)."
        )

    with col2:
        st.markdown("#### Geographic Location")
        locations = comparison_data.get('locations', ['Downtown', 'Suburbs', 'Uptown', 'Waterfront', 'Rural'])
        location = st.selectbox(
            "📍 Location / Neighborhood Tier",
            options=locations,
            index=0,
            help="Geographic zone impacting baseline valuation multiplier."
        )
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Prepare input payload
        input_data = pd.DataFrame([{
            'Area': float(area),
            'Bedrooms': int(bedrooms),
            'Bathrooms': float(bathrooms),
            'Location': str(location),
            'Age': int(age)
        }])
        
        # Predict Button / Real-time computation
        predicted_val = model.predict(input_data)[0]
        price_per_sqft = predicted_val / area
        
        st.markdown(f"""
        <div class='prediction-card'>
            <div class='prediction-title'>Estimated Market Value</div>
            <div class='prediction-value'>${predicted_val:,.2f}</div>
            <div style='margin-top: 10px; font-size: 0.95rem; opacity: 0.85;'>
                Est. Price / Sq Ft: <strong>${price_per_sqft:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.info(f"Model: **{comparison_data.get('best_model', 'XGBoost')}** with R² of **{comparison_data['results'][comparison_data.get('best_model', 'XGBoost')]['R2']}**")

# ----------------- TAB 2: MODEL COMPARISON -----------------
with tab_comparison:
    st.write("### 🏆 Regression Model Comparison & Benchmarks")
    st.caption("Evaluation on 20% holdout test set using R², Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).")
    
    results = comparison_data.get('results', {})
    comp_df = pd.DataFrame(results).T.reset_index()
    comp_df.columns = ['Model Name', 'RMSE ($)', 'MAE ($)', 'R² Score']
    comp_df = comp_df.sort_values(by='R² Score', ascending=False)
    
    col_table, col_chart = st.columns([1, 1], gap="medium")
    
    with col_table:
        st.markdown("#### Benchmark Metrics Table")
        st.dataframe(
            comp_df.style.format({
                'RMSE ($)': '${:,.2f}',
                'MAE ($)': '${:,.2f}',
                'R² Score': '{:.4f}'
            }).highlight_max(subset=['R² Score'], color='#dcfce7'),
            use_container_width=True
        )
        
        best_name = comparison_data.get('best_model')
        st.success(f"Champion Model: **{best_name}** achieved the lowest prediction error and highest variance explained.")
        
    with col_chart:
        st.markdown("#### R² Score Comparison")
        st.bar_chart(
            comp_df.set_index('Model Name')['R² Score'],
            color="#2563eb"
        )

# ----------------- TAB 3: EDA PLOTS -----------------
with tab_eda:
    st.write("### 📊 Exploratory Data Analysis Visualizations")
    st.caption("Visual insights generated during Step 2 & Step 4 of the Capstone project.")
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        p1 = "plots/price_distribution.png"
        if os.path.exists(p1):
            st.image(p1, caption="Figure 1: Target Variable (Price) Distribution", use_container_width=True)
        else:
            st.warning("Distribution plot not found.")
            
    with row1_c2:
        p2 = "plots/area_vs_price.png"
        if os.path.exists(p2):
            st.image(p2, caption="Figure 2: Living Area vs. House Price by Neighborhood", use_container_width=True)
        else:
            st.warning("Area vs Price plot not found.")
            
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        p3 = "plots/bedrooms_bathrooms_vs_price.png"
        if os.path.exists(p3):
            st.image(p3, caption="Figure 3: Room Counts vs Price Distribution", use_container_width=True)
        else:
            st.warning("Room count boxplots not found.")
            
    with row2_c2:
        p4 = "plots/correlation_heatmap.png"
        if os.path.exists(p4):
            st.image(p4, caption="Figure 4: Correlation Matrix of Numeric Attributes", use_container_width=True)
        else:
            st.warning("Correlation plot not found.")

# ----------------- TAB 4: METHODOLOGY -----------------
with tab_pipeline:
    st.write("### 🛠️ Capstone Pipeline Architecture")
    
    st.markdown("""
    This project adheres to the exact specifications in the **Data Science Capstone Project 2 Guide**:
    
    1. **Data Ingestion & Inspection (Step 1)**:
       - Housing dataset loaded containing property metrics (`Area`, `Bedrooms`, `Bathrooms`, `Location`, `Age`).
    2. **Exploratory Data Analysis (Step 2)**:
       - Distribution inspection, feature correlations, and multi-factor boxplots.
    3. **Missing Value Imputation (Step 3)**:
       - Numerical missing values imputed using **Median** (`SimpleImputer`).
       - Categorical missing values imputed using **Mode / Most Frequent** (`SimpleImputer`).
    4. **Outlier Detection & Removal (Step 4)**:
       - Applied the **IQR Method**: computed $IQR = Q_3 - Q_1$, flagging extreme values outside $[Q_1 - 1.5 \\times IQR, Q_3 + 1.5 \\times IQR]$.
    5. **Feature Transformation & Scaling (Step 5 & 6)**:
       - Categorical encoding: **One-Hot Encoding** (`OneHotEncoder`).
       - Numeric scaling: **StandardScaler** to ensure zero-mean and unit variance.
    6. **Model Building & Regularization (Step 7)**:
       - Trained **Linear Regression**, **Ridge** ($L_2$), **Lasso** ($L_1$), **Random Forest**, and **XGBoost**.
    7. **Evaluation Metrics (Step 8)**:
       - Validated models on unseen test data using **$R^2$**, **RMSE**, and **MAE**.
    8. **Interactive Deployment (Bonus)**:
       - Packaged the full pipeline with **Streamlit** for real-time user predictions.
    """)
