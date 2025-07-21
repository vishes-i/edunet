import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
# --- Load Model & Encoder ---
model = pickle.load(open(r"fast_salary_model.pkl", "rb"))
encoder = pickle.load(open(r"fast_label_encoders.pkl", "rb"))

# --- Page Config ---
st.set_page_config(page_title="💼 Salary Predictor", layout="centered")

# --- Sidebar as Navbar ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Prediction", "ℹ️ About"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("💼 Salary Prediction System")
    st.markdown("""
    Welcome to the **Salary Predictor App**!  
    This tool helps estimate whether an individual's income is more or less than 50K based on:
    - Age
    - Education Level
    - Job Title
    - Years of Experience
    - Capital Gain/Loss  
    Use the sidebar to navigate and start predicting! 🎯
    """)

# --- Prediction Page ---
elif page == "📈 Prediction":
    st.title("📊 Salary Prediction Tool")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 70, 30)
            education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
            capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
        with col2:
            job = st.selectbox("Job Title", ["Engineer", "Manager", "Clerk", "Director"])
            experience = st.slider("Years of Experience", 0, 40, 5)
            capital_loss = st.number_input("Capital Loss", 0, 99999, 0)

        submitted = st.form_submit_button("Predict 💰")

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame([{
            "Age": age,
            "Education": education,
            "Job_Title": job,
            "Years_of_Experience": experience,
            "Capital_Gain": capital_gain,
            "Capital_Loss": capital_loss
        }])

        # Apply one-hot encoding (align with training columns)
        input_encoded = encoder.transform(input_data)

        # Predict
        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0]

        # Map to salary range
        income_map = {
            '<=50K': "₹30,000–₹50,000",
            '>50K': "₹60,000–₹1,00,000+"
        }

        st.success(f"🎉 Predicted Salary Category: `{prediction}`")
        st.info(f"💰 Estimated Monthly Salary: **{income_map.get(prediction, 'Unknown')}**")

        # Show probability chart
        st.markdown("### 🔢 Prediction Confidence")
        prob_df = pd.DataFrame({
            "Income": model.classes_,
            "Probability": prob
        })
        st.bar_chart(prob_df.set_index("Income"))

# --- About Page ---
elif page == "ℹ️ About":
    st.title("ℹ️ About this App")
    st.markdown("""
    This salary prediction system uses a **Random Forest model** trained on income-related data  
    to predict whether a person earns more than 50K annually based on various features.

    **Tech Stack:**
    - Python
    - Streamlit
    - scikit-learn
    - Pandas

    📧 Contact: example@example.com
    """)

# --- Footer ---
st.markdown("<hr><center>© 2025 Salary Predictor App</center>", unsafe_allow_html=True)
