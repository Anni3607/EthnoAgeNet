import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
# This model expects a 3-channel (RGB) image input.
model = tf.keras.models.load_model("EthnoAgeNet.h5")

st.title("EthnoAgeNet - Age & Ethnicity Prediction")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded file as a NumPy array of bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode the image in color (1 ensures 3 channels: BGR)
    img = cv2.imdecode(file_bytes, 1)
    
    # Convert BGR image to RGB as most TensorFlow models expect RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the model's expected input size and normalize pixel values
    # The image is resized to 128x128 pixels, and then each pixel value
    # is divided by 255.0 to scale it from [0, 255] to [0.0, 1.0].
    img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
    
    # Expand the dimensions to create a batch of a single image.
    # The model expects input in the shape [batch_size, height, width, channels].
    img_expanded = np.expand_dims(img_resized, axis=0)

    # Make predictions using the loaded model
    pred_age, pred_eth = model.predict(img_expanded)
    
    # Extract predicted age and ethnicity index from the model output
    pred_age = int(pred_age[0][0])
    pred_eth = np.argmax(pred_eth)

    # Map the ethnicity index to a human-readable string
    ethnicity_map = {0:"White", 1:"Black", 2:"Asian", 3:"Indian", 4:"Others"}

    # Display the results on the Streamlit web app
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Age:** {pred_age}")
    st.write(f"**Predicted Ethnicity:** {ethnicity_map[pred_eth]}")
