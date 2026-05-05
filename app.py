import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page settings
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Title and description
st.title("Diabetes Prediction System")
st.markdown("### Early detection of diabetes using Machine Learning")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info(
    "This application predicts whether a person is likely to have diabetes "
    "based on medical inputs using the K-Nearest Neighbors (KNN) algorithm."
)
st.sidebar.markdown("Developed for Healthcare ML Project")

# Input section
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    # Prediction with loading spinner
    with st.spinner("Analyzing data..."):
        result = model.predict(input_data)

    st.subheader("Prediction Result")

    if result[0] == 1:
        st.error("High Risk of Diabetes")
        st.warning("Please consult a doctor for further medical advice.")
    else:
        st.success("Low Risk of Diabetes")
        st.info("Maintain a healthy lifestyle and regular checkups.")

# Footer
st.markdown("---")
st.markdown("Project: Early Prediction of Diabetes using KNN")
