# Import packages

import streamlit as st
import joblib
import pandas as pd

# add title and instructions to the page
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and see how likely they are to purchase.")

# load our model pipeline
model = joblib.load("model.joblib")

# Input elements for customer info
age = st.number_input(
    label = "(1) Enter Customer's Age", 
    min_value = 18,
    max_value = 120,
    value = 35)

gender = st.radio(
    label = "(2) Select Customer's Gender",
    options = ["M", "F"]
)

credit_score = st.number_input(
    label = "(3) Enter Customer's Credit Score",
    min_value = 0,
    max_value = 1000,
    value = 500
)

# Submit option
if st.button("Submit for Prediction"):
    # store our data into a dataframe for prediction
    new_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "credit_score": [credit_score]
    })

    # apply model pipeline to extract the prediction for the new customer
    pred_proba = model.predict_proba(new_data)[0][1]

    # Show the prediction on the user interface
    st.subheader(f"Based on these customer details, the model predicts a probability to purchase of {pred_proba: .0%}")
