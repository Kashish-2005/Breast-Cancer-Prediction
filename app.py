import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model & scaler
model = tf.keras.models.load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")
st.caption("AI-powered detection of Benign vs Malignant tumors")

st.markdown("---")

# ---------- INPUT SECTIONS ----------

st.subheader("📊 Mean Features")
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", value=0.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=0.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.0)
    concavity_mean = st.number_input("Concavity Mean", value=0.0)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.0)

with col2:
    texture_mean = st.number_input("Texture Mean", value=0.0)
    area_mean = st.number_input("Area Mean", value=0.0)
    compactness_mean = st.number_input("Compactness Mean", value=0.0)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.0)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.0)

st.markdown("---")

st.subheader("📈 Standard Error Features")
col3, col4 = st.columns(2)

with col3:
    radius_se = st.number_input("Radius SE", value=0.0)
    perimeter_se = st.number_input("Perimeter SE", value=0.0)
    smoothness_se = st.number_input("Smoothness SE", value=0.0)
    concavity_se = st.number_input("Concavity SE", value=0.0)
    symmetry_se = st.number_input("Symmetry SE", value=0.0)

with col4:
    texture_se = st.number_input("Texture SE", value=0.0)
    area_se = st.number_input("Area SE", value=0.0)
    compactness_se = st.number_input("Compactness SE", value=0.0)
    concave_points_se = st.number_input("Concave Points SE", value=0.0)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.0)

st.markdown("---")

st.subheader("📉 Worst Features")
col5, col6 = st.columns(2)

with col5:
    radius_worst = st.number_input("Radius Worst", value=0.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=0.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.0)
    concavity_worst = st.number_input("Concavity Worst", value=0.0)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.0)

with col6:
    texture_worst = st.number_input("Texture Worst", value=0.0)
    area_worst = st.number_input("Area Worst", value=0.0)
    compactness_worst = st.number_input("Compactness Worst", value=0.0)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.0)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.0)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Cancer Type", use_container_width=True):

    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])

    # 🔑 Apply same scaling as training
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0][0]

    st.markdown("### 🧪 Prediction Result")

    if prediction >= 0.5:
        st.error(f"🛑 **Malignant Tumor Detected**\n\nConfidence: {prediction:.2%}")
    else:
        st.success(f"✅ **Benign Tumor**\n\nConfidence: {(1 - prediction):.2%}")
