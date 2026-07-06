import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration for enterprise deployment
st.set_page_config(
    page_title="RetainAI | Churn Diagnostics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Bold Typography, Structural Border-Cards, and Clean Layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    .main-header {
        background-color: #111217;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-weight: 400;
        margin: 5px 0 0 0;
        font-size: 1rem;
    }
    
    .content-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    .section-title {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 15px;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
    }
    
    div.stButton > button:first-child {
        background-color: #10b981;
        color: white;
        border-radius: 6px;
        width: 100%;
        font-weight: 700;
        height: 50px;
        font-size: 1.1rem;
        border: none;
        letter-spacing: 0.5px;
        transition: background-color 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #059669;
    }
    
    .result-box-stable {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 4px;
    }
    
    .result-box-churn {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 4px;
    }
    
    .result-text-headline {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Top Premium Header Panel (Dark Hero Segment)
st.markdown("""
    <div class='main-header'>
        <h1>RetainAI Systems</h1>
        <p>Enterprise Gradient Boosting Engine for Customer Attrition Risk Management</p>
    </div>
""", unsafe_allow_html=True)

# Load model artifacts securely
@st.cache_resource
def load_pipeline_artifacts():
    return joblib.load("churn_artifacts.pkl")

try:
    artifacts = load_pipeline_artifacts()
    model = artifacts['model']
    encoders = artifacts['encoders']
    feature_names = artifacts['feature_names']
except FileNotFoundError:
    st.error("Operational Error: 'churn_artifacts.pkl' missing from runtime directory.")
    st.stop()

# Restructured Workspace: Inputs organized inside a Bold Tabbed Matrix
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Customer Feature Configuration</div>", unsafe_allow_html=True)

tab_account, tab_financial, tab_services = st.tabs([
    "Core Account Parameters", 
    "Financial & Demographic Profile", 
    "Deployed Services Suite"
])

with tab_account:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        tenure = st.number_input("Tenure (Months Active)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col_a2:
        paperless = st.selectbox("Paperless Billing Status", ["Yes", "No"])
        payment_method = st.selectbox("Payment Routing Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

with tab_financial:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        monthly_charges = st.number_input("Monthly Fixed Charges (USD)", min_value=10.0, max_value=150.0, value=65.0)
        total_charges = st.number_input("Total Cumulative Charges (USD)", min_value=10.0, max_value=9000.0, value=780.0)
        gender = st.selectbox("Gender Identification", ["Female", "Male"])
    with col_f2:
        senior_citizen = st.selectbox("Senior Citizen Classification", ["No", "Yes"])
        partner = st.selectbox("Marital Partner Registered", ["Yes", "No"])
        dependents = st.selectbox("Dependents Registered", ["Yes", "No"])

with tab_services:
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        phone_service = st.selectbox("Voice Line Provision", ["Yes", "No"])
        multiple_lines = st.selectbox("Multi-Line Allocation", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Broadband Infrastructure", ["DSL", "Fiber optic", "No"])
    with col_s2:
        online_security = st.selectbox("Network Security Protocol", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Cloud Backup Allocation", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Hardware Protection Policy", ["No", "Yes", "No internet service"])
    with col_s3:
        tech_support = st.selectbox("Priority Technical Support Escalation", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("IPTV Streaming Service", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("On-Demand Media Streaming", ["No", "Yes", "No internet service"])

st.markdown("</div>", unsafe_allow_html=True)

# High-Visibility Action Button
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("EXECUTE ATTRITION RISK DIAGNOSTICS")
st.markdown("<br>", unsafe_allow_html=True)

# Evaluation Diagnostic Output Block
if analyze_btn:
    # Process user inputs into dictionary mapping target training keys
    raw_inputs = {
        'gender': gender, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    # Build complete array tracking feature order constraint
    encoded_features = []
    for col in feature_names:
        val = raw_inputs[col]
        if col in encoders:
            try:
                val = encoders[col].transform([val])[0]
            except ValueError:
                val = 0
        encoded_features.append(val)
        
    features_array = np.array(encoded_features).reshape(1, -1)
    probability = model.predict_proba(features_array)[0][1]
    
    # Internal retention target parameter configuration
    RETENTION_THRESHOLD = 0.25
    prediction = 1 if probability >= RETENTION_THRESHOLD else 0
    
    # Stark, high-contrast structural cards for results
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>System Diagnostic Output</div>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([2, 1], gap="large")
    
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
                <div class='result-box-churn'>
                    <div class='result-text-headline' style='color: #b91c1c;'>CRITICAL WARNING: HIGH ATTRITION RISK</div>
                    <div style='color: #7f1d1d;'>The account behavioral signature shows critical alignment with customer churn profiles. Immediate defensive retention mitigation is advised.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box-stable'>
                    <div class='result-text-headline' style='color: #047857;'>STABLE SIGNATURE: RETENTION PREDICTED</div>
                    <div style='color: #064e3b;'>The account data vector matches healthy structural retention indicators. Account status is verified secure.</div>
                </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        st.metric(
            label="Calculated Attrition Probability", 
            value=f"{probability * 100:.2f}%",
            delta="Retention Threat Flagged" if prediction == 1 else "Account Status Stable",
            delta_color="inverse" if prediction == 1 else "normal"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)