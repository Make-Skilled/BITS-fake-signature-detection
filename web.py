import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load trained model
MODEL_PATH = "signature_authentication_model.h5"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match model input
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

# Streamlit UI
st.title("🖊 Signature Verification App")
st.write("Upload a signature image to check if it is **real or forged**.")

# File uploader
uploaded_file = st.file_uploader("Upload Signature Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Signature", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)[0]
    confidence_real = prediction[1] * 100
    confidence_forged = prediction[0] * 100

    # Show result
    st.subheader("Prediction:")
    if prediction[1] > prediction[0]:
        st.success(f"✅ The signature is **Real** with {confidence_real:.2f}% confidence.")
    else:
        st.error(f"❌ The signature is **Forged** with {confidence_forged:.2f}% confidence.")

    # Show confidence scores
    st.progress(int(confidence_real if prediction[1] > prediction[0] else confidence_forged))

