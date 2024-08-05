import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your TensorFlow model
model = load_model('riceplantdetectionmodel.h5')

# Define a function to preprocess and predict the image
def predict_image(img):
    img = img.resize((300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0]
    
    # Post-processing of prediction (e.g., get class label from prediction)
    # For example, if your model outputs probabilities for each class:
    class_label = np.argmax(prediction)
    class_names = ['Bacterial blight', 'Brown spot', 'Leaf smut']
    result = class_names[class_label]
    
    return result  # Return the predicted class label

# Streamlit app
st.title("Rice Plant Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image of the rice plant", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Perform prediction
    result = predict_image(img)
    
    # Display the result
    st.write("Classification Result:", result)
