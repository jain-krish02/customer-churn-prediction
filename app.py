import streamlit as st
import pandas as pd
import pickle

model_data = pickle.load(open('customer_churn_model.pkl','rb'))
model = model_data['model']
feature_names = model_data['feature_names']
encoders = pickle.load(open('encoders.pkl','rb'))

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male","Female"])
tenure = st.slider("Tenure (in months)", 0, 72)
monthly = st.number_input("Monthly Charges")

if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure
    }

    df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    df = df[feature_names]

    pred = model.predict(df)

    if pred[0] == 1:
        st.error("High Churn Risk: Customer likely to leave")
    else:
        st.success("Low Churn Risk: Customer likely to stay")