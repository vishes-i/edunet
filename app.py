import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideDown 0.8s ease-out;
    }

    .main-header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
        animation: fadeInUp 0.6s ease-out;
    }

    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }

    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .prediction-label {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem
    }

    @keyframes slideDown {
        0% {
            opacity: 0;
            transform: translateY(-50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
        100% {
            transform: scale(1);
        }
    }

    .stSelectbox > label, .stNumberInput > label, .stRadio > label {
        font-weight: 600;
        color: #333;
    }

    .stButton button {
        background-color: #667eea;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton button:hover {
        background-color: #5a6cdb;
    }

</style>
""", unsafe_allow_html=True)

# Load the trained model
try:
    model = joblib.load(r"C:\Users\adity\PycharmProjects\JupyterProject1\random_forest_regressor_salary_predictor_v1.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Load the original training data to get column names and unique values for encoding
try:
    df_train = pd.read_csv(r"C:\Users\adity\OneDrive\Apps\adult.csv")  # Assuming the original data is available
    df_train.dropna(inplace=True)
    df_train.drop_duplicates(inplace=True)
    df_train.reset_index(inplace=True, drop=True)
    df_train['income_numeric'] = df_train['income'].apply(lambda x: 1 if x == '>50K' else 0)
    df_train["experience"] = df_train["age"] - df_train["educational-num"]

    # Separate features (X) and target (y)
    X_train = df_train.drop(columns=["income", "income_numeric"])

    # Get the list of columns after one-hot encoding the training data
    non_numeric_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
    X_train_encoded = pd.get_dummies(X_train, columns=non_numeric_cols_train, drop_first=True) * 1
    trained_columns = X_train_encoded.columns
except FileNotFoundError:
    st.error("Original data file not found. Cannot prepare data for prediction.")
    st.stop()


# Function to preprocess user input
def preprocess_input(data, trained_columns):
    # Create a DataFrame from the input data
    df_input = pd.DataFrame([data])

    # Handle categorical features using one-hot encoding
    non_numeric_cols_input = df_input.select_dtypes(include=['object', 'category']).columns
    df_input_encoded = pd.get_dummies(df_input, columns=non_numeric_cols_input, drop_first=True) * 1

    # Ensure the order of columns in the input data matches the training data
    # Add missing columns with a value of 0
    missing_cols = set(trained_columns) - set(df_input_encoded.columns)
    for c in missing_cols:
        df_input_encoded[c] = 0

    # Ensure the order of columns is the same
    df_input_encoded = df_input_encoded[trained_columns]

    return df_input_encoded

# Header
st.markdown("""
<div class="main-header">
    <h1>Salary Prediction App</h1>
    <p>Predict your potential salary bracket (>50K or <=50K)</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.header("Enter Your Details")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.selectbox("Workclass", df_train['workclass'].unique())
    education = st.selectbox("Education", df_train['education'].unique())
    marital_status = st.selectbox("Marital Status", df_train['marital-status'].unique())
    occupation = st.selectbox("Occupation", df_train['occupation'].unique())

with col2:
    relationship = st.selectbox("Relationship", df_train['relationship'].unique())
    race = st.selectbox("Race", df_train['race'].unique())
    gender = st.radio("Gender", df_train['gender'].unique())
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Native Country", df_train['native-country'].unique())

# Calculate educational-num and experience from education and age
educational_num_map = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8,
    'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
    'Prof-school': 15, 'Doctorate': 16
}
educational_num = educational_num_map.get(education, 0) # Default to 0 if education not found

# Experience is calculated as age - educational-num
# However, this might result in negative values for younger ages with higher education.
# A more realistic approach might be to use a minimum experience or handle this differently.
# For this example, we'll stick to the calculation for simplicity, but acknowledge this limitation.
experience = age - educational_num

# Capital Gain and Loss - set to 0 as they are not typically user inputs for prediction
capital_gain = 0
capital_loss = 0

# fnlwgt - set to a default or average value, as it's also not a typical user input
# Using the mean from the training data as a placeholder
fnlwgt = df_train['fnlwgt'].mean()


# Prepare the input data as a dictionary
input_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,

    'hours-per-week': hours_per_week,
    'native-country': native_country,
    'experience': experience # Include the calculated experience
}

st.markdown('</div>', unsafe_allow_html=True) # End input-section

# Prediction button
if st.button("Predict Salary Bracket"):
    # Preprocess the input data
    processed_input = preprocess_input(input_data, trained_columns)

    # Make prediction
    prediction = model.predict(processed_input)

    # Interpret the prediction (0 for <=50K, 1 for >50K)
    predicted_bracket = ">50K" if prediction[0] > 0.5 else "<=50K" # Using a threshold of 0.5

    # Display prediction
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("Predicted Salary Bracket:")
    st.markdown(f'<div class="prediction-amount">{predicted_bracket}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
