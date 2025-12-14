import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="World Happiness Score Predictor",
    page_icon="😊",
    layout="wide"
)

# ------------------ LOAD DATA & MODEL ------------------
@st.cache_resource
def load_resources():
    model = pickle.load(open("world_happiness_model.pkl", "rb"))
    encoder = pickle.load(open("country_encoder.pkl", "rb"))
    data = pd.read_csv("2019.csv")
    return model, encoder, data

model, encoder, data = load_resources()

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#ff6159;'>🌍 World Happiness Score Predictor</h1>
    <p style='text-align:center;font-size:17px;'>An ML-based application to estimate happiness using socio-economic factors</p>
    <br>
    """,
    unsafe_allow_html=True
)

# ------------------ CENTERED INPUT SECTION ------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🎛️ Input Parameters")

    country = st.selectbox(
        "Select Country",
        sorted(data['Country or region'].unique())
    )

    gdp = st.slider("GDP per Capita", 0.0, 2.5, 1.0, 0.01)
    social = st.slider("Social Support", 0.0, 2.5, 1.0, 0.01)
    health = st.slider("Healthy Life Expectancy", 0.0, 1.5, 0.8, 0.01)
    freedom = st.slider("Freedom to Make Life Choices", 0.0, 1.5, 0.5, 0.01)
    generosity = st.slider("Generosity", 0.0, 1.0, 0.2, 0.01)
    corruption = st.slider("Perceptions of Corruption", 0.0, 1.0, 0.3, 0.01)

    predict_btn = st.button("🔮 Predict Happiness Score", use_container_width=True)

# ------------------ PREDICTION ------------------
if predict_btn:
    encoded_country = encoder.transform([country])[0]

    features = np.array([[
        encoded_country,
        gdp,
        social,
        health,
        freedom,
        generosity,
        corruption
    ]])

    prediction = model.predict(features)[0]

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")
    st.metric("Predicted Happiness Score", round(prediction, 2))

    # ------------------ MEDIUM SIZE PROBABILITY BAR ------------------
    st.markdown("### 📊 Happiness Probability")
    score_percent = min(max(prediction / 10, 0), 1)

    fig_bar, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["Happiness Level"], [score_percent * 100])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage")
    st.pyplot(fig_bar)

    # ------------------ MEDIUM SIZE FEATURE GRAPH ------------------
    st.markdown("### 📈 Feature Distribution")
    feature_names = [
        "GDP",
        "Social",
        "Health",
        "Freedom",
        "Generosity",
        "Corruption"
    ]

    feature_values = [gdp, social, health, freedom, generosity, corruption]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(feature_names, feature_values)
    ax.set_ylabel("Value")
    ax.set_title("Selected Input Feature Values")
    st.pyplot(fig)

# ------------------ FOOTER ------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center;font-size:14px;'>🚀 Built with Streamlit | Linear Regression | World Happiness Report 2019</p>
    """,
    unsafe_allow_html=True
)
