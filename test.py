import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

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

# Upload an image through the Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Button to trigger classification
    if st.button("Classify"):
        st.write("Classifying...")

        # Make predictions
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)

        predicted_class_index = np.argmax(prediction)

        class_labels = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Eczema', 'Nail Fungus and other nail disorders']

        # Display the prediction result
        st.write(f"Prediction: {class_labels[predicted_class_index]}")
        st.write(f"Confidence: {prediction[0][predicted_class_index]:.2%}")

    #Hello

