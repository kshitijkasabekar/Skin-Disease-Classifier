import streamlit as st
import os
import numpy as np
from keras.preprocessing import image
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from keras.models import load_model
# Load environment variables
load_dotenv()

# Configure GenerativeAI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load skin disease classification model
model_skin_disease = load_model('vgg16_adam.h5')

# Function to preprocess the uploaded image for skin disease classification
def preprocess_image_skin_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to preprocess the uploaded image for the generative model
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to get response from GenerativeAI model
def get_gemini_response(input_prompt, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input_prompt, image[0]])
    return response.text

# Streamlit app
st.set_page_config(page_title="Skin Disease Classifier and Advisor")
st.title("Skin Disease Classifier and Advisor")

# Upload an image through the Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Button to trigger classification and advice
    if st.button("Classify and Advise"):
        # Classification
        img_array = preprocess_image_skin_disease(uploaded_file)
        prediction = model_skin_disease.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        class_labels = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Eczema', 'Nail Fungus and other nail disorders']
        skin_disease_prediction = class_labels[predicted_class_index]

        # Advice
        input_prompt = f"You have been diagnosed with {skin_disease_prediction}. Please provide preliminary treatment suggestions."
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_prompt, image_data)

        # Display the prediction result and advice
        st.subheader("Skin Disease Prediction:")
        st.markdown(f"<p style='font-size: 18px'><strong>{skin_disease_prediction}</strong></p>", unsafe_allow_html=True)

        st.subheader("Preliminary Treatment Suggestions:")
        st.write(response)
