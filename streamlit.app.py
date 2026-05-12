# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# 1. Train a simple ML model
# -----------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("🌱 Iris Flower Prediction App")
st.write("Enter flower measurements to predict species using RandomForest.")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

st.subheader("🔮 Prediction Result")
st.write(f"Predicted Species: **{iris.target_names[prediction]}**")

# -----------------------------
# 3. Probability Graph
# -----------------------------
st.subheader("📊 Prediction Probabilities")
fig, ax = plt.subplots()
ax.bar(iris.target_names, prediction_proba, color=["skyblue","lightgreen","salmon"])
ax.set_ylabel("Probability")
ax.set_ylim(0,1)
st.pyplot(fig)
