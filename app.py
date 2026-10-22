"""Iris Species Classifier - Streamlit app (Module 8B)."""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

LABELS = ["setosa", "versicolor", "virginica"]


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()

st.title("🌸 Iris Species Classifier")
st.write("Set the flower measurements (in cm) and the model predicts the species.")

with st.sidebar:
    st.header("Model info")
    st.write(f"**Model:** {info['model']}")
    st.write(f"**Version:** {info['version']}")
    st.write(f"**Trained:** {info['trained_at']}")
    st.write(f"**Test accuracy:** {info['test_accuracy']:.1%}")
    st.caption(f"Trained on {info['n_train']} samples, tested on {info['n_test']}.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

if st.button("Predict", type="primary"):
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"],
    )
    probs = model.predict_proba(features)[0]
    idx = int(np.argmax(probs))

    st.success(f"Predicted species: **{LABELS[idx].capitalize()}**")
    st.write(f"Confidence: **{probs[idx]:.1%}**")
    st.write("Probability for each species:")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=LABELS))
else:
    st.info("Adjust the sliders, then click **Predict**.")
