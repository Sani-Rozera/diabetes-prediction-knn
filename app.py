import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model/knn_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.title("Diabetes Prediction System")

st.write("Enter patient details:")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data = scaler.transform(data)

    result = model.predict(data)

    if result[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")