import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Portuguese Bank Campaign Prediction")

def user_input_features():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Type of job", ("Admin.", "Blue-collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self-employed", "Services", "Student", "Technician", "Unemployed", "Other"))
    married = st.checkbox("Is married?")
    education = st.radio("Level of education", ("Primary", "Secondary", "Tertiary", "Unknown"))
    default = st.checkbox("Has credit in default?")
    balance = st.number_input("Average yearly balance", value=1000)
    housing = st.checkbox("Has housing loan?")
    loan = st.checkbox("Has personal loan?")  
    prev_contacted = st.checkbox("Contacted prior to current campaign?")
    prev_success = st.checkbox("Subscribed to prior campaign?")
    
    job_map = {
        "Management": "white-collar",
        "Technician": "white-collar",
        "Entrepreneur": "white-collar",
        "Blue-collar": "blue-collar",
        "Unknown": "other",
        "Retired": "unemployed",
        "Admin.": "white-collar",
        "Services": "other",
        "Self-employed": "white-collar",
        "Unemployed": "unemployed",
        "Housemaid": "other",
        "Student": "unemployed"
    }
    job = job_map.get(job, "other")
    
    education = education.lower()
    
    data = {
        "age": age,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "prev_contacted": prev_contacted,
        "prev_success": prev_success,
        "married": married,
    }
    
    job_cat = ["job_blue-collar", "job_other", "job_unemployed", "job_white-collar"]
    education_cat = ["education_primary", "education_secondary", "education_tertiary", "education_unknown"]
    
    for category in job_cat:
        data[category] = False
    for category in education_cat:
        data[category] = False
    
    data["job_" + job] = True
    data["education_" + education] = True
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

model = joblib.load("trained_bank_gbc_model.pkl")

st.subheader("User Input:")
st.write(df)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader("Prediction:")
st.write("Likely to subscribe to term deposit." if prediction[0] else "Not likely subscribe to term deposit.")

st.subheader("Prediction Probability:")
st.write(prediction_proba)
