import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
df = pd.read_csv("Titanic_train.csv")
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # male = 1, female = 0
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, "titanic_model.pkl")
print("✅ Model trained and saved as 'titanic_model.pkl'")


import streamlit as st
import joblib
import numpy as np
@st.cache_resource
def load_model():
    try:
        model = joblib.load("titanic_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'titanic_model.pkl' is in the same folder.")
        return None
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below:")
pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2)
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", min_value=1, max_value=100, value=30)
fare = st.slider("Fare Paid (£)", min_value=0.0, max_value=600.0, value=50.0, step=0.5)
sex_encoded = 1 if sex == "male" else 0
features = np.array([[pclass, sex_encoded, age, fare]])
model = load_model()
if model and st.button("Predict"):
    prediction = model.predict(features)[0]
    outcome = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {outcome}")

    

