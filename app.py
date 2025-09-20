import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("decision_tree_wine_model.pkl")

st.title("🍷 Wine Quality Predictor (Decision Tree)")
st.write("Enter the chemical attributes of a wine sample to predict if it's **Good** or **Not Good**.")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.5f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

if st.button("Predict Quality"):
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be **GOOD QUALITY** (Confidence: {confidence:.2f})")
    else:
        st.error(f"❌ This wine is predicted to be **NOT GOOD QUALITY** (Confidence: {confidence:.2f})")
