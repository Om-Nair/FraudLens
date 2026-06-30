import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import os
from dotenv import load_dotenv
from google import genai

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="FraudLens", layout="wide")

st.title("FraudLens: Explainable Fraud Detection System")
st.caption("AI-assisted fraud investigation prototype combining ML + rules + LLM reasoning")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/finance_fraud_data.csv")

# ----------------------------
# FEATURES
# ----------------------------
features = [
    "Amount",
    "Distance_from_Home",
    "IP_Risk_Score",
    "Avg_Spending_Habit",
    "Is_Weekend",
    "Is_Night_Transaction"
]

X = df[features]

# ----------------------------
# PREPROCESSING + MODEL
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.03, random_state=42)
df["anomaly_score"] = model.fit_predict(X_scaled)

# ----------------------------
# RISK SCORING (UPGRADE 1)
# ----------------------------
def compute_risk(row, anomaly_flag):
    score = 0

    if anomaly_flag == -1:
        score += 2

    if row["Amount"] > row["Avg_Spending_Habit"] * 3:
        score += 1

    if row["Distance_from_Home"] > 50:
        score += 1

    if row["IP_Risk_Score"] > 80:
        score += 2

    if row["Is_Night_Transaction"] == 1:
        score += 1

    if score <= 1:
        return "Low"
    elif score <= 3:
        return "Medium"
    else:
        return "High"

df["risk"] = df.apply(lambda r: compute_risk(r, r["anomaly_score"]), axis=1)

# ----------------------------
# EVIDENCE ENGINE
# ----------------------------
def build_evidence(row):
    evidence = []

    if row["Amount"] > row["Avg_Spending_Habit"] * 3:
        evidence.append("High transaction amount vs normal spending pattern")

    if row["Distance_from_Home"] > 50:
        evidence.append("Transaction far from usual location")

    if row["IP_Risk_Score"] > 80:
        evidence.append("High-risk IP detected")

    if row["Is_Night_Transaction"] == 1:
        evidence.append("Unusual night-time transaction")

    if row["Is_Weekend"] == 1:
        evidence.append("Weekend transaction")

    if len(evidence) == 0:
        evidence.append("No strong anomaly signals detected")

    return evidence

# ----------------------------
# KPI DASHBOARD
# ----------------------------
st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", len(df))
col2.metric("High Risk Cases", len(df[df["risk"] == "High"]))
col3.metric(
    "High Risk %",
    f"{len(df[df['risk']=='High'])/len(df)*100:.2f}%"
)

# ----------------------------
# DATA VIEW
# ----------------------------
st.subheader("Transactions")

st.dataframe(df)

# ----------------------------
# TRANSACTION SELECTION
# ----------------------------
st.subheader("Case Investigation")

idx = st.number_input("Select transaction index", 0, len(df) - 1, 0)

row = df.iloc[idx]

st.write("Selected Transaction")
st.json(row.to_dict())

# ----------------------------
# EVIDENCE
# ----------------------------
evidence = build_evidence(row)

st.subheader("Evidence")
st.write(evidence)

# ----------------------------
# LLM ANALYSIS
# ----------------------------
st.subheader("AI Fraud Analyst Case Note")

if st.button("Generate Investigation Summary"):

    prompt = f"""
You are a senior fraud investigation analyst at a global bank.

Write a structured internal case note.

Evidence:
{evidence}

Include:
- Reason for flag
- Risk interpretation
- Recommended action (monitor, verify, block, etc.)

Keep it concise and professional.
"""

    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.success("Analysis complete")
        st.write(response.text)

    except Exception as e:
        st.error(f"LLM Error: {e}")

# ----------------------------
# OPTIONAL EVALUATION (DO NOT BREAK APP)
# ----------------------------
st.subheader("Model Evaluation (Optional)")

if st.checkbox("Run Evaluation Metrics"):

    df["pseudo_label"] = np.where(
        (df["Amount"] > df["Avg_Spending_Habit"] * 3) |
        (df["IP_Risk_Score"] > 80) |
        (df["Distance_from_Home"] > 50),
        1,
        0
    )

    df["pred_label"] = np.where(df["anomaly_score"] == -1, 1, 0)

    from sklearn.metrics import classification_report, confusion_matrix

    st.write("Confusion Matrix")
    st.write(confusion_matrix(df["pseudo_label"], df["pred_label"]))

    st.write("Classification Report")
    st.text(classification_report(df["pseudo_label"], df["pred_label"]))