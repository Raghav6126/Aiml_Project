import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🔬",
    layout="centered",
)

# Load and cache dataset
@st.cache_data
def load_data():
    """Load the diabetes dataset from a CSV file."""
    return pd.read_csv("diabetes.csv")

# Build and cache model and scaler resource
@st.cache_resource
def build_model():
    """Train the logistic regression model and return the model, scaler, and test accuracy."""
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc

# Initialize model, scaler, and accuracy
model, scaler, accuracy = build_model()

# Main app function
def main():
    st.title("🔬 Diabetes Prediction App")
    st.markdown("Enter patient medical data to predict the risk of diabetes.")

    with st.form("prediction_form"):
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=10, max_value=100, value=33)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 **Diabetes Detected!** (Probability: {proba:.2f})")
        else:
            st.success(f"✅ **No Diabetes Detected.** (Probability: {proba:.2f})")

    st.markdown("---")
    st.subheader("Model Performance")
    st.write(f"Model Accuracy: **{accuracy:.2%}** on test data")

# Run the app
if __name__ == "__main__":
    main()
