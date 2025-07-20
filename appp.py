import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load or create sample data
@st.cache_data
def load_data():
    data: DataFrame = pd.DataFrame({
        'Experience': [1, 3, 5, 7, 9],
        'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters'],
        'Job_Title': ['Analyst', 'Manager', 'Director', 'Analyst', 'Manager'],
        'Salary': [40000, 60000, 90000, 50000, 70000]
    })
    return data

data = load_data()

# Preprocess data
df = pd.get_dummies(data, columns=['Education', 'Job_Title'], drop_first=True)
X = df.drop('Salary', axis=1)
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model for reuse
joblib.dump(model, r"C:\Users\adity\PycharmProjects\JupyterProject1\random_forest_regressor_salary_predictor_v1.pkl")

# Streamlit UI
st.title("Salary Prediction System")

# User Inputs
experience = st.slider('Years of Experience', 0, 20, 1)
education = st.selectbox('Education Level', ['Bachelors', 'Masters', 'PhD'])
job_title = st.selectbox('Job Title', ['Analyst', 'Manager', 'Director'])

# Prepare user input for prediction
input_df = pd.DataFrame([[experience, education, job_title]], columns=['Experience', 'Education', 'Job_Title'])
input_encoded = pd.get_dummies(input_df, columns=['Education', 'Job_Title'])

# Ensure same columns as training
for col in X.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X.columns]  # Order columns

# Load model and predict
model = joblib.load(r'C:\Users\adity\PycharmProjects\JupyterProject1\random_forest_regressor_salary_predictor_v1.pkl')
prediction = model.predict(input_encoded)[0]

st.subheader("Predicted Salary")
st.write(f"${prediction:,.2f}")
