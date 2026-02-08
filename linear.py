import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction using Linear Regression")

st.title("🩺 Diabetes Progression Prediction")
st.write("Linear Regression model using the built-in Diabetes dataset")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target, diabetes.feature_names

X, y, feature_names = load_data()

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

model, X_test, y_test, y_pred = train_model(X, y)

# -----------------------------
# Evaluation metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R² Score):** {r2:.2f}")

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📈 Visual Analysis")

# True vs Predicted plot
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "k--", lw=2)
ax1.set_xlabel("True Values")
ax1.set_ylabel("Predicted Values")
ax1.set_title("True vs Predicted Values")
ax1.grid(True)
st.pyplot(fig1)

# Feature vs Predicted (BMI feature)
bmi_index = 2  # BMI is feature index 2

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.scatter(X_test[:, bmi_index], y_pred, alpha=0.7)
ax2.set_xlabel("BMI (Feature)")
ax2.set_ylabel("Predicted Diabetes Progression")
ax2.set_title("BMI vs Predicted Values")
ax2.grid(True)
st.pyplot(fig2)

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("🧮 Predict Diabetes Progression")

st.write("Enter values for all 10 features:")

user_input = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"**Predicted Diabetes Progression Value:** {prediction[0]:.2f}")
