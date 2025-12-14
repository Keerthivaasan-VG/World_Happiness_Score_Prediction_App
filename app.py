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
    <h1 style='text-align:center;color:#ff6159;'>🌍 World Happiness Score Prediction</h1>
    <p style='text-align:center;font-size:18px;'>Predict a country's happiness score using socio-economic indicators</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR ------------------
st.sidebar.header("🎛️ Input Parameters")

country = st.sidebar.selectbox(
    "Select Country",
    sorted(data['Country or region'].unique())
)

gdp = st.sidebar.slider("GDP per Capita", 0.0, 2.5, 1.0, 0.01)
social = st.sidebar.slider("Social Support", 0.0, 2.5, 1.0, 0.01)
health = st.sidebar.slider("Healthy Life Expectancy", 0.0, 1.5, 0.8, 0.01)
freedom = st.sidebar.slider("Freedom to Make Life Choices", 0.0, 1.5, 0.5, 0.01)
generosity = st.sidebar.slider("Generosity", 0.0, 1.0, 0.2, 0.01)
corruption = st.sidebar.slider("Perceptions of Corruption", 0.0, 1.0, 0.3, 0.01)

# ------------------ ENCODE & PREDICT ------------------
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

# ------------------ MAIN RESULT ------------------
st.markdown("## 🎯 Predicted Happiness Score")
st.metric(label="Happiness Score", value=round(prediction, 2))

# ------------------ PROBABILITY BAR ------------------
st.markdown("## 📊 Happiness Level")

score_percent = min(max(prediction / 10, 0), 1)

fig_bar, ax = plt.subplots()
ax.barh(["Happiness Probability"], [score_percent * 100])
ax.set_xlim(0, 100)
ax.set_xlabel("Percentage")

st.pyplot(fig_bar)

# ------------------ COMPARISON GRAPH ------------------
st.markdown("## 📈 Feature Contribution Visualization")

feature_names = [
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom",
    "Generosity",
    "Corruption"
]

feature_values = [gdp, social, health, freedom, generosity, corruption]

fig, ax = plt.subplots()
ax.bar(feature_names, feature_values)
ax.set_ylabel("Value")
ax.set_title("Input Feature Distribution")
plt.xticks(rotation=30)

st.pyplot(fig)

# ------------------ FOOTER ------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>🚀 Built with Streamlit | Linear Regression Model | World Happiness Report</p>
    """,
    unsafe_allow_html=True
)
