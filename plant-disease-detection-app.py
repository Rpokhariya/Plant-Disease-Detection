import streamlit as st
import pickle
import json
from PIL import Image
import numpy as np
import zipfile
import os

# --- 1. One-time setup: Unzip the model if not already extracted ---

# Define the paths for the zip file and the files to be extracted
MODEL_ZIP_PATH = 'plant_disease_model.zip'
MODEL_PKL_PATH = 'model.pkl'
CLASS_INDICES_PATH = 'class_indices.json'

# Check if the model files are already extracted
if not (os.path.exists(MODEL_PKL_PATH) and os.path.exists(CLASS_INDICES_PATH)):
    # Show a message while unzipping
    with st.spinner('Model is being loaded for the first time, please wait...'):
        # Unzip the file
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall('.')
    st.success('Model loaded successfully!')


# --- 2. Load the model and class indices (cached for performance) ---

# This function loads the model and indices and caches them so it doesn't reload on every interaction.
@st.cache_resource
def load_model_and_indices():
    # Loading the pre-trained model
    with open(MODEL_PKL_PATH, 'rb') as f:
        model = pickle.load(f)
    # Loading the class names
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_model_and_indices()


# --- 3. Prediction Functions (No changes needed here) ---

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# --- 4. Streamlit App Interface (No changes needed here) ---

st.title('🌱 Plant Disease Detector')

uploaded_image = st.file_uploader("Upload an image...", type=('jpg', 'jpeg', 'png'))

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img)

    with col2:
        if st.button('Detect Disease'):
            # Get the prediction
            prediction = predict_image_class(model, uploaded_image, class_indices)
            # Display the result
            st.success(f'Prediction: {str(prediction)}')
else:
    st.warning("Please upload an image to start the prediction.")