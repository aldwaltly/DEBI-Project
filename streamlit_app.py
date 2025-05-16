import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load data
@st.cache_data
def load_data():
    df=pd.read_csv(r"D:\Study\DEPI\Project\DEBI-Project\diabetes_prediction_dataset_cleaned.csv")
    return df

df = load_data()
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

categorical_cols = ['gender', 'smoking_history']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Model selection
model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"])

if model_choice == "Logistic Regression":
    model = pickle.load(open(r"D:\Study\DEPI\Project\DEBI-Project\models\logistic_regression.pkl", "rb"))
elif model_choice == "Random Forest":
    model = pickle.load(open(r"D:\Study\DEPI\Project\DEBI-Project\models\random_forest.pkl", "rb"))
else:
    model = pickle.load(open(r"D:\Study\DEPI\Project\DEBI-Project\models\xgboost.pkl", "rb"))


st.title("Diabetes Prediction App")

# User input interface
st.header("Enter Patient Information")

user_input = {}
for col in numerical_cols:
    user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

for col in categorical_cols:
    user_input[col] = st.selectbox(f"{col}", options=X[col].unique().tolist())

input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"**Prediction:** {'Diabetic' if pred else 'Non-Diabetic'}")
    st.write(f"**Probability of Diabetes:** {prob:.2f}")
    st.write("**Model Used:**", model_choice)
    #now we want to get the accuracy of the model
    accuracy = model.score(X, y)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
