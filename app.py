
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("./audio_classifier2.h5")

# Function to preprocess the .flac audio file
def preprocess_audio(file_path):
    # Load the audio file as mono and with a sample rate of 22050 Hz
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # Check if the audio has been loaded correctly
    if len(audio) == 0:
        raise ValueError("Loaded audio is empty.")
    
    # Extract Mel spectrogram features (shape will be 128 x time_steps)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Ensure the spectrogram has a fixed shape (pad or truncate to 128x157)
    fixed_length = 157  # Adjust to match the model's expected input
    if log_mel_spec.shape[1] < fixed_length:
        # Pad with zeros if shorter
        pad_width = fixed_length - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if longer
        log_mel_spec = log_mel_spec[:, :fixed_length]
    
    # Add channel dimension
    log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
    return log_mel_spec

# Streamlit UI
st.title("Audio Classification App")
st.write("Upload a `.flac` file to classify as bonafide or spoof.")

uploaded_file = st.file_uploader("Choose a .flac file", type=None)

if uploaded_file is not None:
    if not uploaded_file.name.endswith(".flac"):
        st.error("Invalid file format. Please upload a `.flac` file.")
    elif uploaded_file.size == 0:
        st.warning("The uploaded file is empty. Please upload a valid `.flac` file.")
    else:
        with open("temp_audio.flac", "wb") as f:
            f.write(uploaded_file.read())

        try:
            # Process the uploaded file
            mel_spec = preprocess_audio("temp_audio.flac")
            mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension

            # Print to verify shape
            st.write(f"Processed audio shape: {mel_spec.shape}")

            # Predict with the model
            prediction = model.predict(mel_spec)
            predicted_class = np.argmax(prediction, axis=1)
            class_labels = {0: "spoof", 1: "bonafide"}

            # Display the results
            st.markdown("<span style='color:red; font-weight:bold;'>Prediction</span>", unsafe_allow_html=True)
            st.write(f"Class: {class_labels[predicted_class[0]]}")
            st.write(f"Confidence: {prediction[0][predicted_class[0]]:.2f}")
        except Exception as e:
            st.error(f"Error processing audio: {e}")
