import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_churn_model.json")
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, explainer 

preprocessor, model, explainer = load_artifacts()

st.title("🏦 Retail Bank Churn Risk & Explainability Engine")
st.markdown("Predict customer attrition and analyze individual risk drivers via TreeSHAP.")

# Sidebar inputs
st.sidebar.header("Customer Profile")
total_trans_ct = st.sidebar.slider("Total Transaction Count (12m)", 10, 140, 45)
total_trans_amt = st.sidebar.slider("Total Transaction Amount ($)", 500, 18000, 2000)
revolving_bal = st.sidebar.slider("Total Revolving Balance ($)", 0, 2500, 500)
total_ct_chng = st.sidebar.slider("Change in Trans Count (Q4 vs Q1)", 0.0, 3.5, 0.6)
relationship_count = st.sidebar.slider("Total Products Held", 1, 6, 2)

# Build single-row DataFrame with default values for remaining features
input_dict = {
    'Customer_Age': 45, 'Gender': 'M', 'Dependent_count': 2,
    'Education_Level': 'Graduate', 'Marital_Status': 'Married',
    'Income_Category': '$60K - $80K', 'Card_Category': 'Blue',
    'Months_on_book': 36, 'Total_Relationship_Count': relationship_count,
    'Months_Inactive_12_mon': 2, 'Contacts_Count_12_mon': 3,
    'Credit_Limit': 5000.0, 'Total_Revolving_Bal': revolving_bal,
    'Avg_Open_To_Buy': 4500.0, 'Total_Amt_Chng_Q4_Q1': 0.75,
    'Total_Trans_Amt': total_trans_amt, 'Total_Trans_Ct': total_trans_ct,
    'Total_Ct_Chng_Q4_Q1': total_ct_chng, 'Avg_Utilization_Ratio': 0.1
}
input_df = pd.DataFrame([input_dict])

# Inference
X_processed = preprocessor.transform(input_df)
prob = model.predict_proba(X_processed)[0, 1]

# Display Risk Score
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Risk Assessment")
    st.metric("Churn Probability", f"{prob:.1%}")
    if prob >= 0.50:
        st.error("⚠️ High Attrition Risk - Immediate Retention Outreach Required")
    else:
        st.success("✅ Stable Account - Low Risk")

# Explainability
with col2:
    st.subheader("Local SHAP Attribution")
    feature_names = preprocessor.get_feature_names_out()
    shap_vals = explainer(X_processed)
    shap_vals.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_vals[0], max_display=6, show=False)
    st.pyplot(fig)