import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ──────────────────────────────
# PAGE CONFIG
# ──────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧪",
    layout="wide"
)

# ──────────────────────────────
# APPLY CUSTOM CSS
# ──────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ──────────────────────────────
# LOAD DATA AND MODEL
# ──────────────────────────────
data = pd.read_csv("data.csv")
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop(['diagnosis'], axis=1)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ──────────────────────────────
# SIDEBAR INPUT SLIDERS
# ──────────────────────────────
st.sidebar.header("🧬 Cell Nuclei Measurements")

slider_labels = [
    ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"), ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"), ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"), ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"), ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"), ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

input_data = {}
for label, key in slider_labels:
    input_data[key] = st.sidebar.slider(
        label,
        min_value=float(0),
        max_value=float(X[key].max()),
        value=float(X[key].mean())
    )

# ──────────────────────────────
# HEADER
# ──────────────────────────────
st.title("🩺 Smart Cytology: Breast Cancer Predictor")
st.markdown("""
Welcome to **Smart Cytology** — a diagnostic assistant that analyzes **breast cell characteristics** to predict the status of the sample:

<span style="color: white; background-color: #01DB4B; padding: 0.2em 0.5em; border-radius: 0.5em;">🟢 Benign</span>  
<span style="color: white; background-color: #ff4b4b; padding: 0.2em 0.5em; border-radius: 0.5em;">🔴 Malignant</span>

Use the sliders to input measurements derived from **cytological tests**.
""", unsafe_allow_html=True)



# ──────────────────────────────
# LAYOUT COLUMNS
# ──────────────────────────────
col1, col2 = st.columns([3, 2])

# ──────────────────────────────
# LEFT: GAUGE CHART (INSTEAD OF RADAR)
# ──────────────────────────────
with col1:
    st.subheader("📈 Malignancy Score")

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probabilities = model.predict_proba(input_scaled)[0]

    # Dynamic bar color: green if <50%, else red
    bar_color = "#01DB4B" if probabilities[1] < 0.5 else "#ff4b4b"
    
    # Dynamic threshold color: green if <50%, else red
    threshold_color = "#01DB4B" if probabilities[1] < 0.5 else "#ff4b4b"

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilities[1] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Malignancy Score (%)", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.75,
                'value': probabilities[1] * 100
            }
        }
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

# ──────────────────────────────
# RIGHT: PREDICTION + PROBABILITIES
# ──────────────────────────────
with col2:
    st.subheader("🔬 Prediction Result")

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.markdown("<div class='diagnosis benign'>🟢 Benign Cell Cluster</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='diagnosis malignant'>🔴 Malignant Cell Cluster</div>", unsafe_allow_html=True)

    st.write("")

    if prediction == 0:
        st.markdown("**✅ This indicates a benign (non-cancerous) tumor.** These cells usually resemble normal cells and are less aggressive.")
    else:
        st.markdown("**⚠️ This indicates a malignant (cancerous) tumor.** These cells tend to grow and spread more aggressively.")

    st.markdown(f"**Confidence (Benign)**: `{probabilities[0]:.4f}`")
    st.markdown(f"**Confidence (Malignant)**: `{probabilities[1]:.4f}`")

    st.markdown("---")
    st.subheader("💡 Did You Know?")
    st.markdown("""
    Breast cancer detected early has a **99% survival rate** in localized stages.  
    Regular screenings and accurate diagnostics are critical for early detection.
    """)


# ──────────────────────────────
# FOOTER
# ──────────────────────────────
st.markdown("""
---
🛡️ *This tool is designed to assist healthcare professionals, not to replace them.*  
*Always consult a medical expert for diagnosis and treatment.*

🎯 *Model Accuracy: 97%*
""")
