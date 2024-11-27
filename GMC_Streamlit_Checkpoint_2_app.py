import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

# Load model
loaded_pipeline = joblib.load('model.pkl')

# Load categories
cat_dict= joblib.load("cat_dict.pkl")

st.title("Predict if a person has a bank account")
st.markdown("---")

st.subheader("Enter your data")

# Country
country= st.selectbox(" Country", options=cat_dict["country"])

# Year
year= st.radio("Year", key= "year", options=[2016,2017,2028])

# Location type
location_type= st.selectbox("Location Type", options=cat_dict["location_type"])

# Cellphone acces
cellphone_access= st.radio("Cellphone acces", key="cellphone_acces", options=["Yes", "No"])

# Household Size
household_size= st.slider("Household Size", 1, 20, 1)

# Age of respondent
age_of_respondent= st.slider("Age of respondent", 16, 100, 1)

# Gender of respondent
gender_of_respondent= st.radio("Gender of respondnet", key="gender_of_respondet", options=["Female", "Male"])

# Relationship with head of household
relationship_with_head= st.selectbox("Relationship with head", options=cat_dict["relationship_with_head"])

# Marital status
marital_status= st.selectbox("Marital status", options=cat_dict["marital_status"])

# Education level
education_level= st.selectbox("Education level", options=cat_dict["education_level"])

# Job type
job_type= st.selectbox("Job type", options=cat_dict["job_type"])



btn= st.button("Predict")

if btn:
    # Format input
    input_variables= np.array([country, year, location_type, cellphone_access, household_size,\
                       age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type]).reshape(1,-1)
    col_names= ["country", "year", "location_type", "cellphone_access", "household_size", "age_of_respondent",\
                "gender_of_respondent", "relationship_with_head", "marital_status", "education_level", "job_type"]
    input_df= pd.DataFrame(input_variables, columns= col_names)
    
    # Predict
    prediction= loaded_pipeline.predict(input_df)

    # Display result
    if prediction:
        st.success("This person has a bank account")
    elif not prediction:
        st.info("This person does not have a bank account")
    else:
        st.error("Something wen wrong")
