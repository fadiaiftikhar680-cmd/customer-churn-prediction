from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# --- MODELS LOADING ---
try:
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("SUCCESS: Model and Scaler loaded perfectly via joblib!")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# --- DATA VALIDATION MODELS ---
class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str 

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# --- PREPROCESSING HELPER FUNCTION ---
def preprocess_input(df_input: pd.DataFrame) -> np.ndarray:
    df = df_input.copy()
    
    # TotalCharges handle karein
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True), errors='coerce').fillna(0.0)
    
    # customerID handle karein
    df['customerID'] = df['customerID'].astype('category').cat.codes
    
    # Custom Mappings
    mapping_dict = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {
            'Electronic check': 0, 
            'Mailed check': 1, 
            'Bank transfer (automatic)': 2, 
            'Credit card (automatic)': 3
        }
    }
    
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
        
    feature_order = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    df = df[feature_order]
    return scaler.transform(df)

# --- ENDPOINTS ---

@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "healthy", "message": "API is online and models are ready to serve."}

@app.get("/model-info", tags=["Utility"])
def model_info():
    return {
        "model_name": type(model).__name__,
        "features_count": 20,
        "expected_features": [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
    }

@app.post("/predict", tags=["Prediction"])
def predict_single(data: CustomerData):
    try:
        df = pd.DataFrame([data.model_dump()])
        scaled_data = preprocess_input(df)
        
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Top Risk Factors Logic
        feature_order = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = {feature_order[i]: float(importances[i]) for i in range(len(feature_order))}
            if 'customerID' in feat_imp: del feat_imp['customerID']
            top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:3]
            top_risk_factors = [f"{feat} ({round(imp*100, 2)}% impact)" for feat, imp in top_features]
        else:
            top_risk_factors = ["Contract (High Impact)", "Tenure (Medium Impact)", "MonthlyCharges (Low Impact)"]
            
        return {
            "prediction": "Churn" if prediction == 1 else "Not Churn",
            "churn_probability": f"{round(float(probability) * 100, 2)}%",
            "top_risk_factors": top_risk_factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single prediction failed: {str(e)}")

@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(batch_data: BatchCustomerData):
    try:
        data_list = [cust.model_dump() for cust in batch_data.customers]
        df = pd.DataFrame(data_list)
        scaled_data = preprocess_input(df)
        
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1]
        
        results = []
        for i in range(len(df)):
            results.append({
                "customerID": data_list[i]["customerID"],
                "prediction": "Churn" if predictions[i] == 1 else "Not Churn",
                "churn_probability": f"{round(float(probabilities[i]) * 100, 2)}%"
            })
        return {"batch_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")