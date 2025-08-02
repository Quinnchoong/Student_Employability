# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# 📋 PAGE CONFIGURATION
st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")

# 📋 CUSTOM CSS
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}
html, body, [class*="css"] {
    font-size: 14px;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# 📋 LOAD MODEL & SCALER
@st.cache_resource
def load_model():
    try:
        model = joblib.load("employability_predictor.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.error("⚠️ Model or scaler file not found in the directory.")
    st.stop()

# 📋 HEADER IMAGE
try:
    image = Image.open("business_people.jpeg")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.warning("📷 Header image not found. Skipping...")

# 📋 TITLE
st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>", unsafe_allow_html=True)
st.markdown("Please provide the required input features below:")

# 📋 INPUT FEATURES
feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

# 📋 FORM LAYOUT
col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    inputs['GENDER'] = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", index=1)
    inputs['GENERAL_APPEARANCE'] = st.slider("General Appearance (1-5)", 1, 5, 3)
    inputs['GENERAL_POINT_AVERAGE'] = st.number_input("GPA (0.0 - 4.0)", 0.0, 4.0, 3.0, 0.01)
    inputs['MANNER_OF_SPEAKING'] = st.slider("Manner of Speaking (1-5)", 1, 5, 3)

with col2:
    inputs['PHYSICAL_CONDITION'] = st.slider("Physical Condition (1-5)", 1, 5, 3)
    inputs['MENTAL_ALERTNESS'] = st.slider("Mental Alertness (1-5)", 1, 5, 3)
    inputs['SELF-CONFIDENCE'] = st.slider("Self-Confidence (1-5)", 1, 5, 3)
    inputs['ABILITY_TO_PRESENT_IDEAS'] = st.slider("Ability to Present Ideas (1-5)", 1, 5, 3)

with col3:
    inputs['COMMUNICATION_SKILLS'] = st.slider("Communication Skills (1-5)", 1, 5, 3)
    inputs['STUDENT_PERFORMANCE_RATING'] = st.slider("Performance Rating (1-5)", 1, 5, 3)
    inputs['NO_SKILLS'] = st.radio("Has No Skills?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    inputs['Year_of_Graduate'] = st.number_input("Year of Graduation", 2019, 2025, 2022)

# 📋 Convert Inputs to DataFrame
input_df = pd.DataFrame([inputs])[feature_columns]

# 📋 Prediction Function
def predict_employability(model, scaler, input_df):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[0]
    return prediction[0], prediction_proba

# 📋 Predict Button
if st.button("Predict"):
    pred, proba = predict_employability(model, scaler, input_df)

    if pred == 1:
        st.success("🎉 The student is predicted to be **Employable**!")
        st.balloons()
    else:
        st.warning("⚠️ The student is predicted to be **Less Employable**.")

    st.info(f"📈 Probability of being Employable: **{proba[1]*100:.2f}%**")
    st.info(f"📉 Probability of being Less Employable: **{proba[0]*100:.2f}%**")

# 📋 Footer
st.markdown("---")
st.caption("""
Disclaimer: This prediction model is for research and informational purposes only.  
Version 1.0, © 2025 CHOONG MUH IN | Last updated: August 2025  
Developed by Ms. CHOONG MUH IN (TP068331)
""")
