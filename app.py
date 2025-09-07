import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("EthnoAgeNet.h5")

st.title("EthnoAgeNet - Age & Ethnicity Prediction")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # This line is crucial: cv2.imdecode(..., 1) reads the image in color (3 channels).
    img = cv2.imdecode(file_bytes, 1)
    
    # This line converts the image from BGR (OpenCV's default) to RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # This line resizes and normalizes the image.
    img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
    
    # This adds the batch dimension.
    img_expanded = np.expand_dims(img_resized, axis=0)

    # Predictions
    pred_age, pred_eth = model.predict(img_expanded)
    pred_age = int(pred_age[0][0])
    pred_eth = np.argmax(pred_eth)

    ethnicity_map = {0:"White",1:"Black",2:"Asian",3:"Indian",4:"Others"}

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Age:** {pred_age}")
    st.write(f"**Predicted Ethnicity:** {ethnicity_map[pred_eth]}")
