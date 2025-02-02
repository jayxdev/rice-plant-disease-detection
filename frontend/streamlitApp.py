import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure the model file is in the same directory as this script
model_path = os.path.join(os.path.dirname(__file__), 'riceplantdetectionmodel.h5')

# Load your TensorFlow model
model = load_model(model_path)

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
    # Display the uploaded image with a fixed width
    img = Image.open(uploaded_file)
    
    # Create two columns with custom widths
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("## Classification Result:")
        result = predict_image(img)
        st.write(f"### **{result}**")

# Add custom CSS to center-align the text
st.markdown(
    """
    <style>
    .stImage {
        margin-left: auto;
        margin-right: auto;
    }
    .stText {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
