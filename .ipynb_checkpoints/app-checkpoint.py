import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🧬", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("Predict the likelihood of diabetes based on health metrics.")

st.sidebar.header("📌 Choose Input Method")
input_mode = st.sidebar.radio("Select input type:", ["Manual Entry", "Upload CSV File"])

# Required feature order
feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# --- Manual Input Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Patient Data")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("🔍 Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        confidence = probability if prediction == 1 else 1 - probability

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"**Diabetes Detected** 😟\n\nConfidence: {confidence:.2%}")
        else:
            st.success(f"**No Diabetes** 🎉\n\nConfidence: {confidence:.2%}")

        # Download result
        result_text = f"""
Diabetes Prediction Result
--------------------------
Prediction: {'Diabetes Detected' if prediction == 1 else 'No Diabetes'}
Confidence: {confidence:.2%}

Input Summary:
- Pregnancies: {pregnancies}
- Glucose: {glucose}
- Blood Pressure: {blood_pressure}
- Skin Thickness: {skin_thickness}
- Insulin: {insulin}
- BMI: {bmi}
- Diabetes Pedigree Function: {dpf}
- Age: {age}
"""
        result_bytes = result_text.encode("utf-8")
        st.download_button("⬇️ Download Result", result_bytes, file_name="diabetes_prediction.txt", mime="text/plain")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with the required columns", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            if all(col in df.columns for col in feature_cols):
                predictions = model.predict(df[feature_cols])
                probabilities = model.predict_proba(df[feature_cols])[:, 1]

                threshold = 0.55
                adjusted_predictions = (probabilities > threshold).astype(int)
    
                # Display prediction results
                df["Prediction"] = ["Diabetes" if p == 1 else "No Diabetes" for p in adjusted_predictions]
                df["Confidence"] = [f"{prob:.2%}" for prob in probabilities]
                # df["Prediction"] = ["Diabetes" if p == 1 else "No Diabetes" for p in predictions]
                # df["Confidence"] = [f"{(prob if pred==1 else 1 - prob):.2%}" for pred, prob in zip(predictions, probabilities)]

                st.subheader("📊 Prediction Results")
                st.dataframe(df[feature_cols + ["Prediction", "Confidence"]])

                # Download button
                result_csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Full Result", result_csv, file_name="diabetes_predictions.csv", mime="text/csv")
            else:
                st.warning(f"Your file must include these columns:\n{', '.join(feature_cols)}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
