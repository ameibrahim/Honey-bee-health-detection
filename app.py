import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

# Load the model
model = 'new__model.h5'
model = tf.keras.models.load_model(model)

# Preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))  # Resize to the same size as used in training
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Define class labels
# class_labels = ['cancer', 'chickenpox', "cowpox", "healthy", "measles", "monkeypox"]

# Streamlit interface
st.title(f":orange[Bee Health Detection]")
st.write("This application predicts the health status of honey bees.")

col1, col2, col3 = st.columns(3)

col1.image("images/bee1.jpg")
col2.image("images/bee2.jpg")
col3.image("images/bee3.jpeg")

st.write("Here are the following classifcations available")

code = '''
    0. --> varoa, small hive beetles
    1. --> ant problems
    2. --> varoa hive beetles
    3. --> healthy
    4. --> hive being robbed
    5. --> missing queen
'''
st.code(code, language="markdown")


st.write("To predict, you need to upload an image of a bee to test the following conditions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join('uploads', uploaded_file.name)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(image)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(preprocessed_img)
    classes_x = np.argmax(prediction,axis=1)

    classes = [ "varoa, small hive beetles", "ant problems", "varoa hive beetles", "healthy", "hive being robbed", "missing queen"]  

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.title(f"Prediction: :orange[{classes[classes_x[0]]}]")