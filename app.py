import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # Make sure the file is in the same folder
    return df

df = load_data()

# Preprocess the data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# App UI
st.title("🔬 Diabetes Prediction App")
st.markdown("Enter patient medical data to predict the risk of diabetes.")

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 10, 100)

    submitted = st.form_submit_button("Predict")

# Run prediction
if submitted:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 **Diabetes Detected!** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ **No Diabetes Detected.** (Probability: {proba:.2f})")

# Show accuracy
st.markdown("---")
st.subheader("Model Performance")
acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{acc:.2%}** on test data")
