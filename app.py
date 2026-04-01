import streamlit as st
import pandas as pd
import pickle

model_data = pickle.load(open('customer_churn_model.pkl','rb'))
model = model_data['model']
feature_names = model_data['feature_names']
encoders = pickle.load(open('encoders.pkl','rb'))

st.title("Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on key features.")

gender = st.selectbox("Gender", ["Male","Female"])
tenure = st.slider("Tenure (in months)", 0, 72)
monthly = st.number_input("Monthly Charges")

if st.button("Predict"):
    if monthly == 0:
        st.warning("⚠️ Monthly charges are usually greater than 0. Prediction may be unreliable.")

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
    prob = model.predict_proba(df)[0][1]

    st.write(f"Churn Probability: {prob*100:.1f}%")

    if pred[0] == 1:
        st.error("High Churn Risk: Customer is likely to leave")
    else:
        st.success("Low Churn Risk: Customer is likely to stay")
