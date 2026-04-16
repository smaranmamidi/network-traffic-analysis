import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ----------------------------
# Title
# ----------------------------
st.title("🚨 Network Traffic Analysis Dashboard")
st.write("Detect whether network traffic is NORMAL or ATTACK using AutoEncoder")

# ----------------------------
# Load Model & Scaler
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("autoencoder_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_model()

# ----------------------------
# Threshold (your value)
# ----------------------------
threshold = 8.347576186226653e-06

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("demo_input", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    # ----------------------------
    # Preprocessing
    # ----------------------------
    df.columns = df.columns.str.strip()
    df = df.fillna(0)
    df = df.loc[:, ~df.columns.duplicated()]

    X = df.values
    X_scaled = scaler.transform(X)

    # ----------------------------
    # Prediction
    # ----------------------------
    recon = model.predict(X_scaled, verbose=0)
    mse = np.mean((X_scaled - recon)**2, axis=1)

    results = ["🚨 ATTACK" if e > threshold else "✅ NORMAL" for e in mse]
    df["Prediction"] = results
    df["Error"] = mse

    # ----------------------------
    # Show Results
    # ----------------------------
    st.subheader("🔍 Prediction Results")
    st.dataframe(df.head())

    # ----------------------------
    # Summary
    # ----------------------------
    attack_count = results.count("🚨 ATTACK")
    normal_count = results.count("✅ NORMAL")

    st.subheader("📈 Traffic Summary")
    st.write(f"🚨 Attacks Detected: {attack_count}")
    st.write(f"✅ Normal Traffic: {normal_count}")

    # ----------------------------
    # Visualization
    # ----------------------------
    st.subheader("📊 Traffic Distribution")
    st.bar_chart(pd.Series(results).value_counts())

    # ----------------------------
    # Error Distribution
    # ----------------------------
    st.subheader("📉 Reconstruction Error Distribution")
    st.line_chart(mse)