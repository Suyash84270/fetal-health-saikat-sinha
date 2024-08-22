%%writefile app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the model
with open('gbcl_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data
x_train = pd.read_csv('x_train.csv')

# Streamlit app
st.title("Exploratory Data Analysis and Prediction")

st.write("## Dataset Overview")
st.write(x_train)

# Visualize the data
st.write("## Histograms of Features")

# Generate histograms for each column
for column in x_train.columns:
    st.write(f"### {column}")
    fig, ax = plt.subplots()
    ax.hist(x_train[column], bins=10, color='skyblue', edgecolor='black')
    st.pyplot(fig)

# Prediction Section
st.write("## Make Predictions")

# User input for predictions
input_data = {}
for column in x_train.columns:
    input_data[column] = st.number_input(f"Input {column}", value=float(x_train[column].mean()))

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")
