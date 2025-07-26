import streamlit as st
import pickle
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Load feature list
with open('models/features.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Available models
model_options = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Random Forest": "models/model.pkl",
    "K-Nearest Neighbors": "models/KNN.pkl",
    "XGBoost": "models/XGBoost.pkl",
    "Gradient Boosting": "models/Gradient_Boosting.pkl"
}

# Streamlit UI
st.set_page_config(page_title="Heart Disease Predictor")
st.title("❤️ Heart Disease Prediction App")


with open('./models/Logistic_Regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Sidebar EDA option

    
# Form inputs
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", ["male", "female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting ECG", ["normal", "stt abnormality", "lv hypertrophy"])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", ["True", "False"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# One-hot encode inputs
input_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak,
    "ca": ca,

    "sex_male": 1 if sex == "male" else 0,
    "cp_atypical angina": 1 if cp == "atypical angina" else 0,
    "cp_non-anginal": 1 if cp == "non-anginal" else 0,
    "cp_asymptomatic": 1 if cp == "asymptomatic" else 0,

    "fbs_True": 1 if fbs == "True" else 0,
    "restecg_stt abnormality": 1 if restecg == "stt abnormality" else 0,
    "restecg_lv hypertrophy": 1 if restecg == "lv hypertrophy" else 0,

    "exang_True": 1 if exang == "True" else 0,
    "slope_flat": 1 if slope == "flat" else 0,
    "slope_downsloping": 1 if slope == "downsloping" else 0,

    "thal_fixed defect": 1 if thal == "fixed defect" else 0,
    "thal_reversible defect": 1 if thal == "reversible defect" else 0
}

# Add missing columns
for col in model_features:
    if col not in input_dict:
        input_dict[col] = 0

# Ensure column order
input_df = pd.DataFrame([input_dict])[model_features]

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.subheader("🩺 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ Likely Heart Disease (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ No Heart Disease Detected (Confidence: {1 - proba:.2f})")
