import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

df = pd.read_csv(os.path.join(base_dir, "data", "heart_large_cleaned.csv"))
# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    page_icon="❤️"
)

# ======================
# LOAD MODEL
# ======================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(base_dir, "model", "lr_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "model", "scaler.pkl"))
feature_names = joblib.load(os.path.join(base_dir, "model", "features.pkl"))

# ======================
# CUSTOM STYLE (🔥 key upgrade)
# ======================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.title("❤️ AI-Powered Heart Disease Predictor")
st.caption("Predict risk using machine learning with medical feature analysis")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("🧠 Model Info")
st.sidebar.write("""
- Model: Logistic Regression  
- Optimized for: **Low False Negatives**  
- Focus: Early detection  
""")

# ======================
# INPUT FORM (🔥 major UX upgrade)
# ======================
with st.container():

    st.markdown("## 🧑‍⚕️ Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 0, 100, 50)

        if age < 29:
            st.info("Model trained on adult population (29+). Interpret results cautiously.")
        elif age > 80:
            st.info("Very high age values may be underrepresented in training data.")
        sex = st.selectbox("Sex", ["Male", "Female"])
        trestbps = st.slider("Resting BP",
            int(df.trestbps.min()),
            int(df.trestbps.max()),
            int(df.trestbps.median())
        )

    with col2:
        chol = st.slider("Cholesterol",
            int(df.chol.min()),
            int(df.chol.max()),
            int(df.chol.median())
        )
        thalch = st.slider("Max Heart Rate",
            int(df.thalch.min()),
            int(df.thalch.max()),
            int(df.thalch.median())
        )
        oldpeak = st.slider("Oldpeak",
            float(df.oldpeak.min()),
            float(df.oldpeak.max()),
            float(df.oldpeak.median())
        )

    with col3:
        fbs = st.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
        exang = st.selectbox("Exercise Angina", [0, 1])
        slope = st.selectbox("Slope", ["flat", "upsloping"])

    cp = st.selectbox("Chest Pain Type", [
        "typical angina", "atypical angina", "non-anginal"
    ])

    restecg = st.selectbox("Rest ECG", [
        "normal", "st-t abnormality"
    ])

# ======================
# PREDICT BUTTON
# ======================
if st.button("🚀 Analyze Risk"):

    # Input processing
    input_data = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
    }])

    input_data["cp_atypical angina"] = 1 if cp == "atypical angina" else 0
    input_data["cp_non-anginal"] = 1 if cp == "non-anginal" else 0
    input_data["restecg_normal"] = 1 if restecg == "normal" else 0
    input_data["slope_flat"] = 1 if slope == "flat" else 0
    input_data["slope_upsloping"] = 1 if slope == "upsloping" else 0

    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.divider()

    # ======================
    # RESULT DASHBOARD
    # ======================
    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("## 🩺 Diagnosis")

        if prediction == 1:
            st.error("⚠️ HIGH RISK DETECTED")
        else:
            st.success("✅ LOW RISK")

        st.metric("Risk Probability", f"{prob:.2f}")

        if prob < 0.3:
            st.success("🟢 Low Risk Zone")
        elif prob < 0.6:
            st.warning("🟡 Moderate Risk Zone")
        else:
            st.error("🔴 High Risk Zone")

    with colB:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "#00FF9C"},
                    {'range': [30, 60], 'color': "#FFD700"},
                    {'range': [60, 100], 'color': "#FF4B4B"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # FEATURE EXPLANATION
    # ======================
    st.markdown("## 📊 Why this prediction?")

    coeffs = model.coef_[0]

    feature_contrib = pd.DataFrame({
        "Feature": feature_names,
        "Value": input_data.iloc[0],
        "Weight": coeffs
    })

    feature_contrib["Contribution"] = feature_contrib["Value"] * feature_contrib["Weight"]
    feature_contrib = feature_contrib.sort_values(by="Contribution", ascending=False)

    top = feature_contrib.head(5)

    colC, colD = st.columns(2)

    with colC:
        st.dataframe(top)

    with colD:
        st.bar_chart(top.set_index("Feature")["Contribution"])

    st.caption("Model optimized to reduce missed diagnoses (false negatives).")