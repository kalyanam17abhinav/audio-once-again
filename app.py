import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("./audio_classifier2.h5")

# Function to preprocess the .flac audio file
def preprocess_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=22050)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    # Aggregate features (mean of each MFCC coefficient across frames)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# Streamlit UI
st.title("Audio Classification App")
st.write("Upload a `.flac` file to classify as bonafide or spoof.")

uploaded_file = st.file_uploader("Choose a .flac file", type=["flac"])

if uploaded_file is not None:
    with open("temp_audio.flac", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process the uploaded file
    mel_spec = preprocess_audio("temp_audio.flac")
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)   # Add batch dimension

    # Predict with the model
    prediction = model.predict(mel_spec)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = {0: "spoof", 1: "bonafide"}

    st.write("Prediction:")
    st.write(f"Class: {class_labels[predicted_class[0]]}")
    st.write(f"Confidence: {prediction[0][predicted_class[0]]:.2f}")
