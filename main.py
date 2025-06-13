import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # Optional if you scaled your features

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🧠 Diabetes Prediction App")
st.markdown("Enter your health metrics or upload a CSV file to predict diabetes.")

st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

st.header("1️⃣ Manual Input")

with st.form("input_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
    glucose = st.number_input("Glucose", 0.0, 200.0)
    blood_pressure = st.number_input("Blood Pressure", 0.0, 150.0)
    skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0)
    insulin = st.number_input("Insulin", 0.0, 900.0)
    bmi = st.number_input("BMI", 0.0, 70.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 120, step=1)
    submit = st.form_submit_button("🔍 Predict")

if submit:
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age]])
    # Apply scaling if you used it in training
    user_data_scaled = scaler.transform(user_data)

    probability = model.predict_proba(user_data_scaled)[0][1]
    prediction = 1 if probability >= threshold else 0
    result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
    confidence = probability if prediction == 1 else 1 - probability

    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{confidence:.2%}**")

st.header("2️⃣ Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            input_data = df[required_columns]
            input_scaled = scaler.transform(input_data)

            predictions_proba = model.predict_proba(input_scaled)[:, 1]
            predictions = (predictions_proba >= threshold).astype(int)

            df["Prediction"] = np.where(predictions == 1, "Diabetes Detected", "No Diabetes Detected")
            df["Confidence"] = np.where(predictions == 1, predictions_proba, 1 - predictions_proba)
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2%}")

            st.success("✅ Predictions complete!")
            st.dataframe(df)

            csv_result = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv_result, "diabetes_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
