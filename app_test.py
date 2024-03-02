import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import tempfile
import os

# Load the trained model
model = tf.keras.models.load_model('vgg16_adam.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit app
st.title("Skin Disease Classifier")

# Define endpoint for uploading image
@st.cache(allow_output_mutation=True)
def classify_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    class_labels = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Eczema', 'Nail Fungus and other nail disorders']
    return class_labels[predicted_class_index]

# Display the uploaded image and classification result
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Button to trigger classification
    if st.button("Classify"):
        st.write("Classifying...")
        result = classify_image(uploaded_file)
        st.markdown(f"<p style='font-size: 24px'><strong>Prediction: {result}</strong></p>", unsafe_allow_html=True)
