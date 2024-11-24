import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model
MODEL_FILE = 'diabetes_model.pkl'

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    model_loaded = True
else:
    model_loaded = False

# App title
st.title('Diabetes Prediction')

# App description
st.write("""
This is a simple web app to predict whether a person has diabetes. 
Please input the required values and click on the 'Predict' button.
""")

# Function to make predictions
def predict_diabetes(features):
    return model.predict([features])

# User inputs for prediction
st.sidebar.header('User Input Features')
age = st.sidebar.number_input('Age', min_value=10, max_value=100, step=1)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, step=1)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=20, max_value=200, step=1)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, step=1)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)

# Predict button
if st.button('Predict'):
    if model_loaded:
        user_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        prediction = predict_diabetes(user_data)
        if prediction[0] == 1:
            st.success('The model predicts that you **have diabetes**.')
        else:
            st.success('The model predicts that you **do not have diabetes**.')
    else:
        st.error('Model file not found. Please ensure the `diabetes_model.pkl` file is in the correct directory.')
