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

# import streamlit as st
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # Load the trained model
# model = load_model("./audio_classifier2.h5")

# # Function to preprocess the .flac audio file
# def preprocess_audio(file_path):
#     try:
#         # Load the audio file
#         audio, sr = librosa.load(file_path, sr=22050)

#         # Debugging: Print the audio and sample rate
#         print(f"Audio shape: {audio.shape}")
#         print(f"Sample rate: {sr}")

#         # Extract MFCC features
#         mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        
#         # Debugging: Print the MFCC shape
#         print(f"MFCC shape: {mfcc.shape}")
        
#         # Aggregate features (mean of each MFCC coefficient across frames)
#         mfcc_scaled = np.mean(mfcc.T, axis=0)

#         # Normalize features (if required by your model)
#         mfcc_scaled = (mfcc_scaled - np.mean(mfcc_scaled)) / np.std(mfcc_scaled)

#         return mfcc_scaled
#     except Exception as e:
#         print(f"Error processing audio file: {e}")
#         return None

# # Streamlit UI
# st.title("Audio Classification App")
# st.write("Upload a `.flac` file to classify as bonafide or spoof.")

# uploaded_file = st.file_uploader("Choose a .flac file", type=["flac"])

# if uploaded_file is not None:
#     # Save the uploaded file to a temporary location
#     with open("temp_audio.flac", "wb") as f:
#         f.write(uploaded_file.read())
    
#     # Process the uploaded file
#     mel_spec = preprocess_audio("temp_audio.flac")
    
#     if mel_spec is not None:
#         # Reshape the features to match model input requirements
#         mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
#         mel_spec = np.expand_dims(mel_spec, axis=0)   # Add batch dimension

#         # Debugging: Print the shape of mel_spec before prediction
#         print(f"Mel spec shape before prediction: {mel_spec.shape}")

#         # Predict with the model
#         prediction = model.predict(mel_spec)
#         predicted_class = np.argmax(prediction, axis=1)
#         class_labels = {0: "spoof", 1: "bonafide"}

#         st.write("Prediction:")
#         st.write(f"Class: {class_labels[predicted_class[0]]}")
#         st.write(f"Confidence: {prediction[0][predicted_class[0]]:.2f}")
#     else:
#         st.write("Error: Unable to process the audio file.")
