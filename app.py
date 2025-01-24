import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Loading the saved model
model = load_model('cifar10_model.keras')

# CIFAR-10 class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

st.title("CIFAR-10 Image Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (32, 32))  # Resize to CIFAR-10 size
    st.image(img, channels="RGB", caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.expand_dims(img.astype('float32') / 255.0, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: **{predicted_class}**")
