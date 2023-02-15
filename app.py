import streamlit as st
import numpy as np
import joblib as jl
# Initialize your trained model
model = jl.load('best_model.pkl')

#Define the layout of the web application
st.title("Heart Attack Risk Prediction")
st.markdown("Enter the following information to predict your risk of a heart attack.")

# Define the user input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure", min_value=1, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol", min_value=1, max_value=1000, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=1, max_value=300, value=150)
exng = st.selectbox("Exercise Induced Angina", [0, 1])
#oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)
#slp = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
caa = st.number_input("Number of Major Vessels Colored by Flourosopy", min_value=0, max_value=4, value=0)
#thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Define the predict button
if st.button("Predict"):
    # Preprocess the user input
    sex = 1 if sex == "Male" else 0
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa]])

    # Make a prediction on the user input
    prediction = model.predict(input_data)

    # Display the prediction to the user
    if prediction == 0:
        st.markdown("### Result: **Low Risk** of a Heart Attack")
    else:
        st.markdown("### Result: **High Risk** of a Heart Attack")
