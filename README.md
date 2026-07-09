# 📊 Customer Churn Prediction System

An end-to-end Machine Learning web application built to predict subscription-based customer churn, explain risk factors, and provide a seamless interface for non-technical business stakeholders and retention teams.

---

## 🎯 1. Business Problem Statement & Context
Subscription-based companies lose valuable revenue when customers churn. This system addresses this challenge by:
1. Identifying high-risk customers before they leave.
2. Explaining the **top driving risk factors** so retention teams know exactly how to intervene.
3. Providing an intuitive interface for single and batch data processing.

---

## 📋 2. Dataset Description
The model is trained on the standard **Telco Customer Churn Dataset** (~7,000 rows, 21 columns). It contains a mix of categorical and numerical features:
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services Signed Up For:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Customer Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable:** `Churn` (Yes/No - indicating if the customer left within the last month).

*Note: Class imbalance (~27% churn rate) and missing white spaces in `TotalCharges` were successfully handled during data preparation.*

---

## 📈 3. Model Comparison Table
During the development phase, multiple machine learning architectures were trained and evaluated using appropriate evaluation metrics (prioritizing Recall/F1-Score due to class imbalance):

| Model Name | Accuracy | Precision | Recall (Class 1) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **80.2%** | **65.4%** | **54.1%** | **59.2%** | **Selected (Deployed)** |
| Logistic Regression | 79.5% | 63.1% | 55.3% | 58.9% | Evaluated |
| Decision Tree Classifier | 72.8% | 50.2% | 51.0% | 50.6% | Evaluated |

*Reason for Selection:* Random Forest was chosen because it offered the best balance between precision and recall, along with stable feature importance calculations for model explanation.

---

## 🏗️ 4. System Architecture Diagram

Below is the operational workflow of the complete ecosystem:

```text
+-----------------------+              +------------------------+              +-----------------------+
|                       |  HTTP POST   |                        |  Joblib Load  |                       |
|   Streamlit UI        | -----------> |   FastAPI Backend      | ------------> |  ML Model (.pkl)      |
|   (Frontend Service)  | <----------- |   (Prediction API)     | <------------ |  Scaler / Encoders    |
|                       |  JSON Res    |                        |  Inference    |                       |
+-----------------------+              +------------------------+              +-----------------------+
           ^                                       |
           |                                       v
   [User Interfaces]                       [Data Preprocessing]
   - Home & Health Check                   - Type Conversions (TotalCharges)
   - Single Manual Entry                   - Explicit Categorical Encoding
   - Bulk CSV Batch Upload                 - StandardScaler Scaling