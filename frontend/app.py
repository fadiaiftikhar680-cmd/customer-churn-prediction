import streamlit as st  
import requests
import pandas as pd     
import json

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# BACKEND URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("📊 Customer Churn Prediction System")
st.markdown("---")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Home & Health", "Single Prediction", "Batch Prediction (CSV File)"])

# ==============================================================================
# 1. HOME & HEALTH MODE
# ==============================================================================
if app_mode == "Home & Health":
    st.subheader("Business Context & System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### API Health Status")
        try:
            health_res = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
            st.success(f"🟢 Backend Status: {health_res['status'].upper()}")
            st.write(f"💬 Message: {health_res['message']}")
        except Exception as e:
            st.error(f"🔴 Backend Offline: Could not connect to FastAPI at {BACKEND_URL}")
            st.write(f"Error Details: {str(e)}")
            
    with col2:
        st.info("### Model Information")
        try:
            model_res = requests.get(f"{BACKEND_URL}/model-info", timeout=5).json()
            st.success(f"🤖 Model Architecture: **{model_res['model_name']}**")
            st.write(f"🔢 Total Features Used: `{model_res['features_count']}`")
        except Exception as e:
            st.error("🔴 Model info unavailable.")

# ==============================================================================
# 2. SINGLE PREDICTION MODE (Updated with Explanations)
# ==============================================================================
elif app_mode == "Single Prediction":
    st.subheader("👤 Individual Customer Churn Analysis")
    st.write("Enter the customer details below to evaluate churn risk and view driving factors.")
    
    # Form Layout
    with st.form(key="single_customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customerID = st.text_input("Customer ID", value="1234-ABCD")
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1])
            Partner = st.selectbox("Has Partner?", ["Yes", "No"])
            Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
            
        with col2:
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            
        with col3:
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.5)
        with c2:
            TotalCharges = st.text_input("Total Charges ($)", value="785.0")
            
        submit_button = st.form_submit_button(label="🔮 Predict Churn")
        
    if submit_button:
        payload = {
            "customerID": customerID, "gender": gender, "SeniorCitizen": int(SeniorCitizen),
            "Partner": Partner, "Dependents": Dependents, "tenure": int(tenure),
            "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport, "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
            "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
            "MonthlyCharges": float(MonthlyCharges), "TotalCharges": str(TotalCharges)
        }
        
        with st.spinner("Analyzing customer profile..."):
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    probability = result['churn_probability']
                    risk_factors = result.get('top_risk_factors', [])

                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    if prediction == "Churn":
                        st.error(f"### 🔥 Alert: Customer is likely to **CHURN**!")
                        st.metric(label="Churn Probability Score", value=probability)
                        
                        st.warning("#### 📋 Why is this customer at risk? (Model Explanation)")
                        st.write("Model ke analysis ke mutabik yeh top factors customer ko churn ki taraf push kar rahe hain:")
                        for factor in risk_factors:
                            st.write(f"- 🔴 **{factor}**")
                            
                        st.info("💡 **Retention Strategy Recommendation:** Is customer ko loyalty discount, personalized support call, ya long-term contract features upgrade offer karein.")
                    else:
                        st.success(f"### ✅ Great! Customer is **LOYAL** (Not Churn).")
                        st.metric(label="Churn Probability Score", value=probability)
                        
                        st.info("#### 📋 Top Driving Factors for Loyalty:")
                        st.write("In attributes ki wajah se customer business ke sath sustain kar raha hai:")
                        for factor in risk_factors:
                            st.write(f"- 🟢 **{factor}**")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed to backend: {str(e)}")

# ==============================================================================
# 3. BATCH PREDICTION MODE
# ==============================================================================
elif app_mode == "Batch Prediction (CSV File)":
    st.subheader("📁 Bulk Customer Batch Processing")
    st.write("Upload a CSV file containing multiple customer records to perform bulk analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV File Uploaded Successfully!")
            st.write("### Raw Data Preview (First 5 Rows):")
            st.dataframe(df.head(5))
            
            if st.button("🚀 Process Bulk Predictions"):
                required_cols = [
                    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
                ]
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Invalid CSV structure. Missing columns: {missing_cols}")
                else:
                    df_filled = df.fillna("")
                    records = df_filled.to_dict(orient="records")
                    payload = {"customers": records}
                    
                    with st.spinner("Processing batch on server..."):
                        response = requests.post(f"{BACKEND_URL}/predict-batch", json=payload)
                        if response.status_code == 200:
                            batch_results = response.json()["batch_results"]
                            res_df = pd.DataFrame(batch_results)
                            final_df = pd.merge(df, res_df, on="customerID", how="left")
                            
                            st.success("🎉 Batch Processing Completed Successfully!")
                            st.write("### Processed Results View:")
                            st.dataframe(final_df[['customerID', 'prediction', 'churn_probability']].head(20))
                            
                            csv = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Full Predictions CSV File",
                                data=csv,
                                file_name="customer_churn_predictions_output.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Batch prediction endpoint failed: {response.text}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")