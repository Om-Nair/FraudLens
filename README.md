# FraudLens — Explainable Fraud Detection System

FraudLens is an end-to-end machine learning prototype that simulates how financial institutions detect and investigate suspicious transactions. It combines anomaly detection, rule-based reasoning, and large language models to generate explainable fraud investigation reports.

The goal is not just to flag fraud, but to explain *why* a transaction is suspicious in a way that mirrors real fraud analyst workflows.

---

## Problem Statement

Most fraud detection systems output binary labels (fraud / not fraud) or opaque risk scores. In real financial environments, compliance and fraud teams need:

- Clear reasoning behind alerts  
- Interpretable evidence  
- Investigation-ready summaries  

FraudLens addresses this gap by building an explainable, multi-layer fraud detection pipeline.

---

## System Overview

FraudLens follows a multi-layer architecture:

### 1. Data Layer
Transaction dataset containing behavioral and contextual features such as:
- Transaction amount  
- Distance from home  
- IP risk score  
- Spending behavior  
- Time-based signals  

---

### 2. Anomaly Detection Layer
An unsupervised Isolation Forest model identifies unusual transaction patterns without requiring labeled fraud data.

---

### 3. Risk Scoring Layer
A rule-augmented scoring system converts anomaly signals into:
- Low Risk  
- Medium Risk  
- High Risk  

This makes outputs interpretable for decision-making.

---

### 4. Explainability Layer
A rule-based engine converts feature-level anomalies into structured human-readable evidence such as:
- High deviation from spending behavior  
- Geographic inconsistency  
- High-risk IP detection  

---

### 5. LLM Reasoning Layer
A large language model (Gemini) generates structured fraud analyst-style investigation notes based on extracted evidence.

This simulates real internal fraud case documentation used in financial institutions.

---

### 6. UI Layer
A Streamlit dashboard enables:
- Transaction exploration  
- Risk-level filtering  
- Evidence inspection  
- AI-generated fraud case summaries  

---

## Key Features

- Unsupervised fraud detection using Isolation Forest  
- Behavioral feature engineering for transactions  
- Tiered fraud risk scoring (Low / Medium / High)  
- Rule-based explainability layer  
- LLM-generated fraud investigation reports  
- Interactive Streamlit dashboard  
- Optional model evaluation using proxy labels  

---

## Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- Streamlit  
- Google Gemini API  

---

## How It Works

1. Load transaction dataset  
2. Extract behavioral features  
3. Train Isolation Forest anomaly detection model  
4. Generate anomaly flags  
5. Compute risk score using rule-based logic  
6. Extract explainable evidence per transaction  
7. Generate AI-powered investigation summary  
8. Display results in interactive dashboard  

---

## Example Output

### Evidence
- High transaction amount vs normal spending pattern  
- Transaction far from usual location  
- High-risk IP detected  

### AI Investigation Summary
The transaction exhibits multiple high-risk indicators including behavioral and geographic anomalies. The combination of unusual spending magnitude and location deviation suggests potential account compromise. Immediate verification or transaction blocking is recommended.

---

## Model Evaluation (Optional)

Since real fraud labels are not available, proxy labels were created using domain-based heuristics. The anomaly detection model is evaluated using:

- Confusion Matrix  
- Precision / Recall  
- Classification Report  

This provides an approximate performance benchmark for the unsupervised model.

---

## Limitations

- Uses synthetic or static dataset  
- Not trained on real banking fraud data  
- Isolation Forest is unsupervised and not fully calibrated  
- LLM output is assistive, not authoritative  

---

## Future Improvements

- Supervised fraud classification model  
- Real-time transaction streaming pipeline  
- SHAP-based feature explainability  
- Model calibration for production use  
- Deployment as a cloud-based fraud monitoring tool  

---

## Author

Om Nair  
Applied Engineering Sciences — Data Science & Analytics Focus  
Michigan State University  

---

## Project Goal

To simulate how modern financial institutions combine machine learning, rule-based systems, and AI reasoning to support fraud investigation workflows.