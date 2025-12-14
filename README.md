# 🌍 World Happiness Score Predictor

## 📌 About the Project
This repository contains a **Machine Learning–based Streamlit web application** that predicts the **World Happiness Score** of a country using key socio-economic indicators. The prediction is powered by a **Linear Regression model** trained on the **World Happiness Report 2019 dataset**.

The application provides an interactive and professional user interface where users can select a country, adjust economic and social factors, and instantly view the predicted happiness score along with meaningful visualizations.

---

## 🎯 Objective
The main objective of this project is to:
- Understand how different socio-economic factors influence happiness
- Build a regression-based prediction model
- Deploy the model as a user-friendly web application

---

## 🧠 Machine Learning Details
- **Algorithm:** Linear Regression
- **Dataset:** World Happiness Report 2019 (CSV)
- **Target Variable:** Happiness Score
- **Features Used:**
  - Country or Region (encoded)
  - GDP per capita
  - Social support
  - Healthy life expectancy
  - Freedom to make life choices
  - Generosity
  - Perceptions of corruption

---

## 🖥️ Web Application Features
- 🎛️ Country selection using dropdown (from dataset)
- 🎚️ Slider-based input for all socio-economic indicators
- 🎯 Real-time happiness score prediction
- 📊 Probability bar visualization
- 📈 Feature distribution bar chart
- 🎨 Clean and professional UI using Streamlit

---

## 📁 Project Structure
```
├── app.py                     # Streamlit application
├── 2019.csv                   # Dataset
├── world_happiness_model.pkl  # Trained Linear Regression model
├── country_encoder.pkl        # Label encoder for country names
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/world-happiness-predictor.git
cd world-happiness-predictor
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## ☁️ Deployment
This application can be easily deployed using **Streamlit Cloud** by uploading the repository and ensuring all required files are present.

---

## 🎓 Academic Use
This project is suitable for:
- Mini Projects
- Machine Learning Assignments
- Data Science Portfolios
- Streamlit Deployment Practice

---

## 📌 Conclusion
The World Happiness Score Predictor demonstrates how machine learning models can be applied to real-world socio-economic data and deployed as interactive web applications. It combines data preprocessing, regression modeling, and user-friendly visualization into a complete end-to-end ML project.

---

✨ *Built using Python, Scikit-learn, and Streamlit*

