import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📊 Exam Score Predictor")

# Sample Data (Hours vs Scores)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
scores = np.array([35, 40, 50, 55, 65, 70, 80, 90])

# Train Model
model = LinearRegression()
model.fit(hours, scores)

# User Input
study_hours = st.slider("Select hours studied:", 0, 10, 1)

# Prediction
prediction = model.predict([[study_hours]])

# Output
st.write(f"📌 Predicted Score: {prediction[0]:.2f}")

# Extra Info
st.write("This prediction is based on a simple linear regression model.")