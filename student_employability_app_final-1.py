# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page config
st.set_page_config(page_title="🎓 Student Employability Predictor", layout="centered")


# Load model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("employability_predictor.pkl")
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']

        return model, label_encoder, feature_names
    except FileNotFoundError:
        return None, None, None


model, label_encoder, feature_columns = load_model()

if model is None:
    st.error("❌ Model file not found.")
    st.stop()

# Header Image
try:
    image = Image.open("business_people.png")
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.warning("📷 Header image not found.")

# Title
st.markdown("<h2 style='text-align: center;'>🎓 Student Employability Predictor — SVM Model</h2>",
            unsafe_allow_html=True)

# Input form
col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    inputs['GENDER'] = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
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
    inputs['NO_SKILLS'] = st.radio("Has No Skills?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Year_of_Graduate'] = st.number_input("Year of Graduation", 2019, 2025, 2023)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])[feature_columns]


# Predict
def predict(model, input_df):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    return pred, proba


# Button
if st.button("Predict"):
    pred, proba = predict(model, input_df)
    predicted_class = label_encoder.inverse_transform([pred])[0]

    st.markdown("---")
    if predicted_class == "Employable":
        st.success(f"🎉 The student is predicted to be **{predicted_class}**!")
        st.balloons()
    else:
        st.warning(f"⚠️ The student is predicted to be **{predicted_class}**.")

    st.info(f"📈 Probability of being Employable: **{proba[0] * 100:.2f}%**")
    st.info(f"📉 Probability of being Less Employable: **{proba[1] * 100:.2f}%**")

# Footer
st.markdown("---")
st.caption("""
Disclaimer: This prediction model is for research and informational purposes only.  
Version 1.0, © 2025 CHOONG MUH IN | Last updated: August 2025  
Developed by Ms. CHOONG MUH IN (TP068331)
""")