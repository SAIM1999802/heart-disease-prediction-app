import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("heart_disease_data.csv")

# Split Data
X = df.drop(columns='target')
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=23
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Title
st.title("❤️ Heart Disease Prediction System")
st.write("Enter Patient Information")

# HUMAN-FRIENDLY INPUTS

age = st.number_input("Age", 1, 100)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])
cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250)

chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Resting ECG Result", [
    "Normal",
    "ST-T abnormality",
    "Left ventricular hypertrophy"
])
restecg = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"].index(restecg)

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("Oldpeak (ST depression)")

slope = st.selectbox("Slope of peak exercise ST segment", [
    "Upsloping",
    "Flat",
    "Downsloping"
])
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)

ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia", [
    "Normal",
    "Fixed defect",
    "Reversible defect"
])
thal = ["Normal", "Fixed defect", "Reversible defect"].index(thal)


# PREDICTION
if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("You are Healthy ❤️")
    else:
        st.error("Risk Detected ⚠️ Please consult a doctor")