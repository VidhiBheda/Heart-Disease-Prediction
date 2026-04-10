import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ─────────────────────────────────────────
# CLEAN CSS (FIXED)
# ─────────────────────────────────────────
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f5f7fa;
    color: #000000;
}

/* Inputs fix */
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

div[data-testid="stSelectbox"] div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

ul[role="listbox"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Labels */
label {
    color: #000000 !important;
    font-weight: 500;
}

/* Button */
.stButton > button {
   background: linear-gradient(135deg, #EE2A2A 0%, #FF6351 100%);
    color: white;
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #e03e3e;
    transform: scale(1.02);
}

/* Result */
.result-low {
    background: #eafaf1;
    border: 1px solid #2ecc71;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.result-high {
    background: #fdecea;
    border: 1px solid #e74c3c;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

/* Disclaimer */
.disclaimer {
    background: #fff3cd;
    padding: 12px;
    border-radius: 8px;
    color: #856404;
    margin-top: 20px;
}

.app-header {
    background: linear-gradient(135deg, #EE2A2A 0%, #FF6351 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 6px 0;
    color: #ffffff !important;
}
.app-header p {
    font-size: 14px;
    opacity: 0.90;
    margin: 0;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header" style="color:#000000;align:center">
    <h1>🫀 Heart Disease Prediction</h1>
    <p>
        Enter the patient's clinical and lifestyle details below.
        The model will predict the likelihood of heart disease or stroke risk.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORM (FIXED)
# ─────────────────────────────────────────
with st.container(border=True):

    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        age = st.number_input("Age", 20, 90, 45)
    with col3:
        education = st.selectbox("Education", ["Uneducated", "Primary School", "Graduate", "Postgraduate"])

    st.subheader("🚬 Smoking Habits")
    col1, col2 = st.columns(2)
    with col1:
        current_smoker = st.selectbox("Current Smoker", ["No", "Yes"])
    with col2:
        cigs_per_day = st.number_input("Cigarettes per Day", 0, 60, 0)

    st.subheader("🏥 Medical History")
    col1, col2, col3 = st.columns(3)
    with col1:
        bp_meds = st.selectbox("BP Medication", ["No", "Yes"])
    with col2:
        stroke = st.selectbox("Stroke History", ["No", "Yes"])
    with col3:
        hyp = st.selectbox("Hypertension", ["No", "Yes"])

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    st.subheader("🔬 Clinical Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        chol = st.number_input("Cholesterol", 100.0, 600.0, 230.0)
    with col2:
        sys_bp = st.number_input("Systolic BP", 80.0, 300.0, 120.0)
    with col3:
        dia_bp = st.number_input("Diastolic BP", 40.0, 180.0, 80.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with col2:
        heart_rate = st.number_input("Heart Rate", 30, 200, 75)
    with col3:
        glucose = st.number_input("Glucose", 40.0, 400.0, 80.0)

    predict = st.button("🔍 Predict", use_container_width=True)

# ─────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────
def encode():
    return np.array([[
        1 if gender == "Male" else 0,
        age,
        ["Uneducated","Primary School","Graduate","Postgraduate"].index(education),
        1 if current_smoker=="Yes" else 0,
        cigs_per_day,
        1 if bp_meds=="Yes" else 0,
        1 if stroke=="Yes" else 0,
        1 if hyp=="Yes" else 0,
        1 if diabetes=="Yes" else 0,
        chol, sys_bp, dia_bp, bmi, heart_rate, glucose
    ]])

if predict:
    data = encode()
    data = scaler.transform(data)
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    st.subheader("📊 Result")

    if pred == 0:
        st.markdown(f"""
        <div class="result-low">
        <h3>✅ Low Risk</h3>
        <p>No Risk: {prob[0]*100:.1f}% | Risk: {prob[1]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-high">
        <h3>⚠️ High Risk</h3>
        <p>No Risk: {prob[0]*100:.1f}% | Risk: {prob[1]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ This is not medical advice. Consult a doctor.
    </div>
    """, unsafe_allow_html=True)