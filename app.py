import streamlit as st
import joblib
import pandas as pd
from preprocessing import OutlierClipper

pipeline = joblib.load('pipeline.pkl')

st.title("Heart Attack Risk Predictor")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    resting_bp = st.slider("Resting Blood Pressure", 80, 200, 130)
    cholesterol = st.slider("Cholesterol", 100, 600, 245)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    max_hr = st.slider("Max Heart Rate", 60, 210, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.slider("ST Depression", 0.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", [0, 1, 2])
    major_vessels = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    patient = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'chest_pain': chest_pain,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': fasting_bs,
        'resting_ecg': resting_ecg,
        'max_hr': max_hr,
        'exercise_angina': exercise_angina,
        'st_depression': st_depression,
        'st_slope': st_slope,
        'major_vessels': major_vessels,
        'thalassemia': thalassemia
    }])

    result = pipeline.predict(patient)

    if result[0] == 1:
        st.error("High Risk of Heart Attack")
    else:
        st.success("Low Risk of Heart Attack")
