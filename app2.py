import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# Load model & encoder (adjust paths accordingly)
model = joblib.load(r"C:\Users\adity\Downloads\fast_salary_model.pkl")
encoders = joblib.load(r"C:\Users\adity\Downloads\fast_label_encoders.pkl")  # renamed for clarity

# --- Page Setup ---
st.set_page_config(page_title="💼 Salary Dashboard", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .title { color: #1f4e79; font-size: 36px; }
    .section { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .footer { color: #888; font-size: 14px; text-align: center; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "📊 Insights", "ℹ️ About"])

# --- Home Page ---
if page == "🏠 Home":
    st.markdown("<div class='title'>💼 Welcome to the Salary Predictor</div>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    st.markdown("""
    This tool predicts whether a person earns more or less than **₹50,000** based on:
    - Age
    - Job Title
    - Education
    - Capital Gain/Loss
    - Experience

    ➡ Use the sidebar to navigate between sections.
    """)

elif page == "📈 Predict":
    st.markdown("<div class='title'>📈 Salary Prediction</div>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 18, 70, 30)
                education_options = list(encoders["education"].classes_)
                education = st.selectbox("Education", education_options)
                capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
                marital_options = list(encoders["marital-status"].classes_)
                marital_status = st.selectbox("Marital Status", marital_options)

            with col2:
                job_options = list(encoders["occupation"].classes_)
                job = st.selectbox("Job Title", job_options)
                gender_options = list(encoders["gender"].classes_)
                gender = st.selectbox("Gender", gender_options)
                hours_per_week = st.slider("Hours per Week", 1, 100, 40)
                experience = st.slider("Years of Experience", 0, 40, 5)
                capital_loss = st.number_input("Capital Loss", 0, 99999, 0)

            st.markdown("</div>", unsafe_allow_html=True)

        # move this OUTSIDE of the nested block to be accessible below
        submitted = st.form_submit_button("🔮 Predict")

    # ✅ Check if submitted and process input
    if submitted:
        # Build input_df
        input_df = pd.DataFrame([{
            "Age": age,
            "Education": education,
            "Job_Title": job,
            "Capital_Gain": capital_gain,
            "Capital_Loss": capital_loss,
            "Gender": gender,
            "Marital_Status": marital_status,
            "Hours_per_Week": hours_per_week
        }])

        # Encode categorical values
        column_mapping = {
            "Education": "education",
            "Job_Title": "occupation",
            "Gender": "gender",
            "Marital_Status": "marital-status"
        }

        from sklearn.preprocessing import LabelEncoder

        for form_col, encoder_key in column_mapping.items():
            le = encoders.get(encoder_key)

            if le is None:
                st.error(f"❌ Encoder for '{encoder_key}' is missing.")
                st.write("Available encoder keys:", list(encoders.keys()))
                st.stop()

            try:
                if isinstance(le, dict):
                    input_df[form_col] = input_df[form_col].map(le)
                elif isinstance(le, LabelEncoder):
                    if input_df[form_col].iloc[0] not in le.classes_:
                        st.error(f"❌ Invalid value '{input_df[form_col].iloc[0]}' for '{form_col}'.")
                        st.write("Allowed:", list(le.classes_))
                        st.stop()
                    input_df[form_col] = le.transform(input_df[form_col])
                else:
                    st.error(f"❌ Encoder for '{encoder_key}' is not supported.")
                    st.stop()
            except Exception as e:
                st.error(f"⚠️ Error while encoding '{form_col}': {e}")
                st.stop()

        # Rename columns to match model
        input_df.rename(columns={
            "Age": "age",
            "Capital_Gain": "capital-gain",
            "Capital_Loss": "capital-loss",
            "Education": "education",
            "Job_Title": "occupation",
            "Gender": "gender",
            "Marital_Status": "marital-status",
            "Hours_per_Week": "hours-per-week"
        }, inplace=True)

        # Predict
        # Reorder columns to match training
    try:
        input_df = input_df[model.feature_names_in_]
    except AttributeError:
        st.error("❌ Model does not contain `feature_names_in_`. You must manually specify feature order.")
        st.stop()
    except KeyError as e:
        st.error(f"❌ Missing or mismatched feature columns: {e}")
        st.write("Expected columns:", list(model.feature_names_in_))
        st.write("Input columns:", list(input_df.columns))
        st.stop()

        # Predict
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        income_map = dict(zip(model.classes_, ["₹30,000–₹50,000", "₹60,000–₹1,00,000+"]))
        salary = income_map.get(pred, "Unknown")

        st.success(f"🎉 Predicted Income Category: `{pred}`")
        st.info(f"💰 Estimated Monthly Salary: **{salary}**")

        st.markdown("### 🔢 Prediction Probability")
        prob_df = pd.DataFrame({
            "Income": model.classes_,
            "Probability": prob
        })
        st.bar_chart(prob_df.set_index("Income"))

    except Exception as e:
        st.error("🚨 Error during prediction.")
        st.exception(e)

# --- Insights Page ---
elif page == "📊 Insights":
    st.markdown("<div class='title'>📊 Salary Insights</div>", unsafe_allow_html=True)

    # Fake data example
    fake_data = pd.DataFrame({
        "Age": np.random.randint(25, 55, 100),
        "Salary": np.random.randint(30000, 100000, 100)
    })
    st.line_chart(fake_data.set_index("Age"))

# --- About Page ---
elif page == "ℹ️ About":
    st.markdown("<div class='title'>ℹ️ About This App</div>", unsafe_allow_html=True)
    st.markdown("""
    This app predicts salary categories using a machine learning model trained on demographic and job data.

    **Built with:**
    - Python 🐍
    - Streamlit 🚀
    - scikit-learn 📊

    Created by [Your Name].
    """)

# --- Footer ---
st.markdown("<div class='footer'>© 2025 Salary Predictor Dashboard</div>", unsafe_allow_html=True)
