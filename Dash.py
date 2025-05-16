import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly


df = pd.read_csv(r"D:\Study\DEPI\Project\DEBI-Project\Data\ForDashboard.csv")

st.title("Diabetes Prediction App")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Risk Factors"])


if page == "Overview":
    st.header("Diabetes Dataset Overview")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        diabetes_percentage = round((df['diabetes'].mean() * 100), 2)
        st.metric("Diabetes Rate", f"{diabetes_percentage}%")
    with col3:
        avg_age = round(df['age'].mean(), 1)
        st.metric("Average Age", f"{avg_age}")
    

    st.subheader("Sample Data")
    st.dataframe(df.head())
    

    st.subheader("Data Summary")
    st.write(df.describe())

elif page == "Data Exploration":
    st.header("Explore the Dataset")

    plot_type = st.selectbox("Select Plot Type", 
                           ["Age Distribution", "BMI Distribution", "Blood Glucose by Diabetes", 
                            "HbA1c Level by Diabetes", "Feature Correlation"])
    
    if plot_type == "Age Distribution":
        fig = px.histogram(df, x='age', color='diabetes', 
                          title='Age Distribution by Diabetes Status',
                          labels={'diabetes': 'Diabetes Status'},
                          barmode='overlay',
                          color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig)
        st.write("This histogram shows the age distribution for diabetic and non-diabetic patients.")
        
    elif plot_type == "BMI Distribution":
        fig = px.histogram(df, x='bmi', color='diabetes', 
                          title='BMI Distribution by Diabetes Status',
                          labels={'diabetes': 'Diabetes Status'},
                          barmode='overlay',
                          color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig)
        st.write("BMI distribution shows patterns between weight status and diabetes.")
        
    elif plot_type == "Blood Glucose by Diabetes":
        fig = px.box(df, x='diabetes', y='blood_glucose_level', 
                    title='Blood Glucose Level by Diabetes Status',
                    labels={'diabetes': 'Diabetes Status', 'blood_glucose_level': 'Blood Glucose Level'})
        st.plotly_chart(fig)
        st.write("Box plot shows the clear relationship between blood glucose levels and diabetes diagnosis.")
        
    elif plot_type == "HbA1c Level by Diabetes":
        fig = px.box(df, x='diabetes', y='HbA1c_level', 
                    title='HbA1c Level by Diabetes Status',
                    labels={'diabetes': 'Diabetes Status', 'HbA1c_level': 'HbA1c Level'})
        st.plotly_chart(fig)
        st.write("HbA1c levels are a key diagnostic criterion for diabetes.")
        
    elif plot_type == "Feature Correlation":

        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        st.write("This heatmap shows correlations between numerical features.")

elif page == "Risk Factors":
    st.header("Diabetes Risk Factors Analysis")

    st.subheader("Diabetes by Gender")
    gender_diabetes = df.groupby('gender')['diabetes'].mean().reset_index()
    gender_diabetes['diabetes_percentage'] = gender_diabetes['diabetes'] * 100
    
    fig = px.bar(gender_diabetes, x='gender', y='diabetes_percentage',
                title='Diabetes Percentage by Gender',
                labels={'diabetes_percentage': 'Diabetes Percentage (%)', 'gender': 'Gender'},
                color='gender')
    st.plotly_chart(fig)

    st.subheader("Risk Factors")
    risk_factor = st.selectbox("Select Risk Factor", 
                              ["Hypertension", "Heart Disease", "Smoking History"])
    
    if risk_factor == "Hypertension":
        hyper_diabetes = df.groupby('hypertension')['diabetes'].mean().reset_index()
        hyper_diabetes['diabetes_percentage'] = hyper_diabetes['diabetes'] * 100
        
        fig = px.bar(hyper_diabetes, x='hypertension', y='diabetes_percentage',
                    title='Diabetes Percentage by Hypertension',
                    labels={'diabetes_percentage': 'Diabetes Percentage (%)', 
                          'hypertension': 'Has Hypertension'},
                    color='hypertension',
                    color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig)
        
        hyper_count = df['hypertension'].value_counts().reset_index()
        hyper_count.columns = ['hypertension', 'count']
        fig2 = px.pie(hyper_count, values='count', names='hypertension', 
                     title='Distribution of Hypertension',
                     hole=0.4)
        st.plotly_chart(fig2)
        
    elif risk_factor == "Heart Disease":
        heart_diabetes = df.groupby('heart_disease')['diabetes'].mean().reset_index()
        heart_diabetes['diabetes_percentage'] = heart_diabetes['diabetes'] * 100
        
        fig = px.bar(heart_diabetes, x='heart_disease', y='diabetes_percentage',
                    title='Diabetes Percentage by Heart Disease',
                    labels={'diabetes_percentage': 'Diabetes Percentage (%)', 
                          'heart_disease': 'Has Heart Disease'},
                    color='heart_disease',
                    color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig)
        
        heart_count = df['heart_disease'].value_counts().reset_index()
        heart_count.columns = ['heart_disease', 'count']
        fig2 = px.pie(heart_count, values='count', names='heart_disease', 
                     title='Distribution of Heart Disease',
                     hole=0.4)
        st.plotly_chart(fig2)
        
    elif risk_factor == "Smoking History":
        smoke_diabetes = df.groupby('smoking_history')['diabetes'].mean().reset_index()
        smoke_diabetes['diabetes_percentage'] = smoke_diabetes['diabetes'] * 100
        
        fig = px.bar(smoke_diabetes, x='smoking_history', y='diabetes_percentage',
                    title='Diabetes Percentage by Smoking History',
                    labels={'diabetes_percentage': 'Diabetes Percentage (%)', 
                          'smoking_history': 'Smoking History'},
                    color='smoking_history')
        st.plotly_chart(fig)
        
        smoke_count = df['smoking_history'].value_counts().reset_index()
        smoke_count.columns = ['smoking_history', 'count']
        fig2 = px.pie(smoke_count, values='count', names='smoking_history', 
                     title='Distribution of Smoking History',
                     hole=0.4)
        st.plotly_chart(fig2)

