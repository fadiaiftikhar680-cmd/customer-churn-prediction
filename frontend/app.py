import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model and Scaler directly from the frontend/models directory
@st.cache_resource
def load_assets():
    model = joblib.load("frontend/models/churn_model.pkl")
    scaler = joblib.load("frontend/models/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Custom Mapping Dictionary for user input transformation
MAPPING = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 2, 'No': 0, 'No phone service': 1},
    'InternetService': {'Fiber optic': 1, 'DSL': 0, 'No': 2},
    'OnlineSecurity': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'OnlineBackup': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'DeviceProtection': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'TechSupport': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'StreamingTV': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'StreamingMovies': {'Yes': 2, 'No': 0, 'No internet service': 1},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 2, 'Mailed check': 3, 
        'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1
    }
}

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict individual customer churn or upload a batch file for instant analytics.")

tabs = st.tabs(["👤 Single Customer Prediction", "📂 Bulk Batch Prediction"])

# Exact columns order expected by your scaler (with customerID)
SCALER_COLUMNS_ORDER = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

# --- TAB 1: SINGLE PREDICTION ---
with tabs[0]:
    st.subheader("Enter Customer Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen (Age >= 65)", [0, 1])
        Partner = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection Service", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support Service", ["No", "Yes", "No internet service"])

    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

    if st.button("Predict Churn Risk", type="primary"):
        input_data = {
            'customerID': 0, # Dummy numerical value for scaler satisfaction
            'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents,
            'tenure': tenure, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
            'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
        }
        
        # Encoding Categorical inputs using mapping
        encoded_data = {}
        for col, val in input_data.items():
            if col in MAPPING:
                encoded_data[col] = MAPPING[col][val]
            else:
                encoded_data[col] = val
                
        df_features = pd.DataFrame([encoded_data])
        
        # Reorder columns to match fit time exactly
        df_features = df_features[SCALER_COLUMNS_ORDER]
        
        # Scale & Predict directly in app
        scaled_features = scaler.transform(df_features)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        st.write("---")
        if prediction == 1:
            st.error(f"🚨 High Risk Customer! Churn Probability: {probability:.2%}")
        else:
            st.success(f"✅ Loyal Customer. Churn Probability: {probability:.2%}")

# --- TAB 2: BATCH PREDICTION ---
with tabs[1]:
    st.subheader("Upload CSV File for Batch Analytics")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("📋 Raw Data Sample:", raw_df.head(3))
        
        if st.button("Process Batch Predictions"):
            try:
                processed_df = raw_df.copy()
                
                # If uploaded CSV doesn't have customerID, create a dummy one
                if 'customerID' not in processed_df.columns:
                    processed_df['customerID'] = 0
                else:
                    # If it exists but is text, map it temporarily to numbers for scaler
                    processed_df['customerID'] = 0
                    
                for col in MAPPING:
                    if col in processed_df.columns:
                        processed_df[col] = processed_df[col].map(MAPPING[col])
                
                # Align exact column order including customerID
                processed_df = processed_df[SCALER_COLUMNS_ORDER]
                
                scaled_batch = scaler.transform(processed_df)
                predictions = model.predict(scaled_batch)
                probabilities = model.predict_proba(scaled_batch)[:, 1]
                
                raw_df['Churn_Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
                raw_df['Churn_Probability'] = probabilities
                
                st.write("---")
                st.success("🎯 Batch Processing Completed!")
                st.dataframe(raw_df)
                
                # Dashboard analytical metrics
                total_cust = len(raw_df)
                churned_cust = sum(predictions)
                st.metric("Total Customers Evaluated", total_cust)
                st.metric("Predicted Churn Cases", f"{churned_cust} ({churned_cust/total_cust:.1%})")
                
            except Exception as ex:
                st.error(f"Encoding/Scaler Error: Make sure columns match perfectly. Details: {ex}")