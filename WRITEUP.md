# 📑 Executive Short Write-up: Model Limitations & Future Roadmap
**Project:** Customer Churn Prediction System  
**Author:** Fadia Iftikhar

---

## 🛑 1. Current Model Limitations
Industrial machine learning systems operate in dynamic environments. While our Random Forest Classifier performs well, it has the following operational limitations:

1. **Static Historical Data (Batch Training):**
   The model is trained on a snapshot of static historical customer details. It cannot inherently adapt to real-time behavioral drifts or changing market trends unless retrained completely with fresh data.

2. **Lack of Sequential/Temporal Tracking:**
   The features used (like `tenure`, `MonthlyCharges`) represent static states. The model cannot track sequential behavior over time—such as a sudden drop in data usage over the last 3 weeks or a recent series of unresolved customer support complaints.

3. **Absence of Advanced Business Context Features:**
   Crucial operational triggers like customer support interaction logs, sentiment analysis from call records, and competitor pricing benchmarks are currently absent from the dataset.

4. **Class Imbalance Constriction:**
   Despite handling techniques, the dataset holds ~27% churn instances. This slight data skew makes the model more conservative when predicting positive churn classes compared to strong loyal patterns.

---

## 🚀 2. What I Would Do Differently With More Time
If given additional development time and resources, I would implement the following architectural enhancements to transition this project into an enterprise-grade solution:

### A. Advanced Explainable AI (XAI) Integration
Instead of relying on global feature importances, I would integrate **SHAP (SHapley Additive exPlanations)** or **LIME** inside the FastAPI backend. This would generate local waterfall charts for *each individual customer* inside the Streamlit frontend, telling the retention specialist exactly how much individual factors (e.g., $+15\%$ risk due to Fiber Optic service) pushed that specific prediction.

### B. MLOps Automated Pipelines
I would implement an automated retraining pipeline using **MLflow** or **DVC (Data Version Control)**. Whenever new client billing profiles arrive, a pipeline would validate data quality, check for data drift, automatically retrain the model, and swap production versions safely without API downtime.

### C. Feature Engineering & Deep Learning
- **Sequential Features:** I would engineer "delta features" (e.g., ratio of this month's bill to the previous 6 months' average) to detect sudden anomalies in usage.
- **Deep Learning:** I would experiment with Neural Networks or GBDT frameworks like **XGBoost** and **LightGBM**, tuning hyperparameters using Bayesian Optimization to push the F1-Score past 65%.

### D. Advanced Deployment & Security
I would containerize the entire stack using **Docker**, set up an automated **GitHub Actions CI/CD pipeline**, and deploy the microservices into a Kubernetes cluster monitored by Prometheus and Grafana for real-time model telemetry.