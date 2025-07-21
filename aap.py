import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import joblib
import os
import sklearn
import random

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error


# ----------------- Constants ------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR,r"C:\Users\adity\Downloads\fast_salary_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, r"C:\Users\adity\Downloads\fast_label_encoders.pkl")
CSV_PATH = r"C:\Users\adity\OneDrive\Apps\adult.csv"
FEATURE_COLUMNS = [
    'age', 'education', 'occupation', 'hours-per-week', 'gender',
    'capital-gain', 'capital-loss', 'marital-status'
]

model = None
encoders = None
trained = False

df = pd.read_csv(CSV_PATH)  # Load once for schema validation

# ----------------- Train Model ------------------
def train_model():
    global df
    X = df[FEATURE_COLUMNS]
    y = df["income"]

    enc = {}
    for col in X.select_dtypes(include=["object"]).columns:
        enc[col] = LabelEncoder()
        X[col] = enc[col].fit_transform(X[col])
    enc["income"] = LabelEncoder()
    y = enc["income"].fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(enc, ENCODER_PATH)

    mse = mean_squared_error(y_test, clf.predict(X_test))
    rmse = mse ** 0.5

    return clf, enc, accuracy_score(y_test, clf.predict(X_test)), rmse

# ----------------- Load or Train (Safe) ------------------
def safe_load_or_train():
    global model, encoders, trained
    try:
        retrain = False

        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)

            model_version = getattr(model, '_sklearn_version', None)
            current_version = sklearn.__version__
            if model_version and model_version != current_version:
                st.warning(f"Incompatible model version: trained on {model_version}, current is {current_version}")
                retrain = True


            for col in FEATURE_COLUMNS + ['income']:
                if df[col].dtype == 'object' and col not in encoders:
                    st.warning(f"Encoder for '{col}' is missing. Triggering retrain.")
                    retrain = True
                    break
        else:
            retrain = True

        if retrain:
            model, encoders, acc, rmse = train_model()
            trained = True
        else:
            trained = True

    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {e} - Re-training now...")
        model, encoders, acc, rmse = train_model()
        trained = True

safe_load_or_train()

# ----------------- Streamlit UI ------------------
st.title("👋 Welcome to My Porfile")
st.set_page_config(page_title="Salary Prediction System", layout="centered")
st.title("💰Salary Prediction System App")

if trained:
    st.markdown("✅ **Predict Your Salary**")

def user_input():
    st.markdown("### 📝 Further Information")

    if not encoders:
        st.error("❌ Encoders not available. Please retrain the model.")
        return None, None

    required_keys = ['education', 'occupation', 'gender', 'marital-status']
    missing = [key for key in required_keys if key not in encoders]
    if missing:
        st.error(f"❌ Missing encoders for: {', '.join(missing)}. Please check your dataset or retrain the model.")
        return None, None

    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", encoders["education"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)

    input_dict = {
        "age": age,
        "education": encoders["education"].transform([education])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "hours-per-week": hours_per_week,
        "gender": encoders["gender"].transform([gender])[0],
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "marital-status": encoders["marital-status"].transform([marital_status])[0],
    }

    readable_input = {
        "Age": age,
        "Education": education,
        "Occupation": occupation,
        "Hours per Week": hours_per_week,
        "Gender": gender,
        "Capital Gain": capital_gain,
        "Capital Loss": capital_loss,
        "Marital Status": marital_status
    }

    return pd.DataFrame([input_dict]), readable_input

# ----------------- Prediction ------------------
input_df, readable_input = user_input()

if input_df is not None:
    try:
        input_df = input_df[FEATURE_COLUMNS]

        if input_df.isnull().values.any():
            st.error("❌ Some required fields are missing.")
        else:
            try:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
            except AttributeError as err:
                st.error("🚨 Prediction failed due to model incompatibility. Retraining...")
                model, encoders, acc, rmse = train_model()
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]

            label = encoders["income"].inverse_transform([pred])[0]
            confidence = prob[pred] * 100



            st.success(f"💰 Predicted Income Category: `{label}`")

            label = '<=50K' or ">50K"

            # Salary range mapping for randomness
            income_map_random_range = {
                '<=50K': (25000, 49000),
                '>50K': (51000, 150000)
            }

            # Get the salary range for the predicted label
            salary_range = income_map_random_range.get(label, (0, 0))

            # Generate random salary within the range
            predicted_salary = random.randint(*salary_range)

            # Display the predicted salary
            st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")
            st.info(f"🔐 Confidence: `{confidence:.2f}%`")

        st.markdown("### 🧾 Your Inputs")
        st.dataframe(pd.DataFrame([readable_input]))

        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Category": encoders["income"].classes_,
            "Probability": prob
        })
        prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
        prob_df = prob_df.sort_values(by="Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Category"))

    except Exception as e:
        st.error("🚨 Unexpected error during prediction.")
        st.exception(e)



import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="👤 My Portfolio", page_icon="🧑‍💻", layout="centered")

# ---- Load Assets ----
profile_pic = Image.open(r"C:\Users\adity\OneDrive\Documents\WhatsApp Image 2025-06-23 at 16.25.47_88df29d7.jpg")  # Replace with your image
name = "Vishesh Kumar Prajapati"
description = "🚀 Aspiring Data Scientist | Python Enthusiast | Machine Learning Explorer"
email = "visheshprajapati7920@gmail.com"

# ---- Main Profile ----
st.image(profile_pic, width=150)

st.title(name)
st.write(description)
st.write(f"📧 Email: [{email}](mailto:{email})")

# ---- Social Media Links ----
st.subheader("🌐 Connect with me")
st.write("""
- [LinkedIn](https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- [GitHub](https://github.com/vishes-i)
- [Twitter](https://twitter.com/VisheshVis78914)
""")

# ---- Skills / Projects ----
st.subheader("💼 Projects / Skills")

projects = {
    "Salary Prediction App": "A machine learning app that predicts salary using Random Forest.",
    "Data Cleaning Tool": "Automated tool for missing value and outlier detection.",
    "Image Classifier": "CNN-based deep learning model for image classification."
}

for project, desc in projects.items():
    st.markdown(f"**{project}**  \n{desc}")




# ----------------- Sidebar ------------------

import streamlit as st
from PIL import Image

# ----------------- Page Config -----------------
st.set_page_config(page_title="💼 Portfolio | Vishesh", page_icon="🧑‍💻", layout="wide")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("👨‍💻 My Profile")

    # Profile Image
    img = Image.open(r"C:\Users\adity\OneDrive\Documents\WhatsApp Image 2025-06-23 at 16.25.47_88df29d7.jpg")  # Replace with your image
    st.image(img, width=150)
    with open(r"C:\Users\adity\OneDrive\Documents\resume.pdf", "rb") as file:
        btn = st.download_button("📄 Download Resume", file, "Vishesh_Resume.pdf")
    # Contact Info
    st.markdown("### 📬 Contact")
    st.markdown("- 📧 [Email](mailto:visheshprajapati7920@gmail.com)")
    st.markdown("- 💼 [LinkedIn](https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.markdown("- 🐙 [GitHub](https://github.com/vishesh-i)")

    # Navigation
    st.markdown("### 📂 Navigation")
    page = st.radio("Go to", ["About Me", "Projects", "Skills", "Contact"])

# ----------------- Main Content -----------------


if page == "About Me":
    st.header("About Me")
    st.write("""
    Hello! I'm **Vishesh Kumar Prajapati**, a passionate Python developer and machine learning enthusiast.
    I love working on data-driven problems and building apps with real-world impact.
    """)
    st.markdown("---")
    st.write("© 2025 Vishesh Kumar Prajapati. Built with ❤️ using Streamlit.")

elif page == "Projects":
    st.header("Projects")
    st.markdown("### 📊 Salary Prediction App")
    st.write("ML model that predicts salary based on demographics and experience.")

    st.markdown("### 🧼 Data Cleaner")
    st.write("Tool to clean, impute, and preprocess raw datasets automatically.")

elif page == "Skills":
    st.header("Skills")
    st.write("""
    - ✅ Python
    - ✅ Machine Learning
    - ✅ Streamlit
    - ✅ Pandas, NumPy
    - ✅ Scikit-learn
    - ✅ Git/GitHub
    """)

elif page == "Contact":
    st.header("Contact Me")
    st.markdown("📧 [Send an Email](mailto:visheshprajapati7920@gmail.com)")
    st.markdown("Or message me on [LinkedIn](https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")




with st.sidebar:
    # Sidebar Header
    st.markdown('<div class="sidebar-header">📊 Prediction Settings</div>', unsafe_allow_html=True)

    # Toggle Options
    show_insights = st.checkbox("📈 Show Career Insights", value=True)
    show_charts = st.checkbox("📉 Show Visualization", value=True)

    # Currency Selector
    currency = st.selectbox("💱 Select Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])

    # Tips Box
    st.markdown("""
        <div class="info-box">
            <h4>💡 Tips for Better Predictions</h4>
            <ul>
                <li>✅ Ensure accurate experience data</li>
                <li>✅ Select the most relevant job title</li>
                <li>✅ Consider location factors</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Model Information
st.sidebar.header("🔍 Model & Encoders")
st.sidebar.subheader("🧠 Model Parameters")

if model:
    st.sidebar.markdown(f"- `n_estimators`: **{getattr(model, 'n_estimators', 'N/A')}**")
    st.sidebar.markdown(f"- `max_depth`: **{getattr(model, 'max_depth', 'N/A')}**")
else:
    st.sidebar.warning("⚠️ Model not loaded.")

# Encoders Display
st.sidebar.subheader("🗂️ Label Encoders")
if encoders:
    for col, le in encoders.items():
        st.sidebar.markdown(f"**{col}**: `{', '.join(le.classes_)}`")
else:
    st.sidebar.warning("⚠️ Encoders not loaded. Please retrain the model.")

# Retrain Button
if st.sidebar.button("🔁 Retrain Model"):
    with st.spinner("⏳ Retraining model..."):
        model, encoders, acc, rmse = train_model()
        st.sidebar.success("✅ Model retrained!")


