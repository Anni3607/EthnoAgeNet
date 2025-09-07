import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Step 1: Define the function that builds the model architecture
# This must match the original model exactly.
def create_model_architecture():
    # Define the input shape. Based on the previous error, the model was trained
    # with 3-channel (color) images, likely 128x128 pixels.
    inputs = Input(shape=(128, 128, 3))
    
    # Add your layers here. This is a placeholder; you must replace it with your
    # actual layers and their parameters (filters, kernel size, activation, etc.).
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x) # The problematic layer from the previous log
    
    # Add more layers as needed from your original model
    
    x = Flatten()(x)
    
    # Assuming two output heads for age and ethnicity
    age_output = Dense(1, activation='linear', name='age_output')(x) # Use linear activation for regression
    ethnicity_output = Dense(5, activation='softmax', name='ethnicity_output')(x) # Use softmax for classification

    model = Model(inputs=inputs, outputs=[age_output, ethnicity_output])
    return model

# Step 2: Manually build the model and load the weights
# This bypasses the broken architecture in the H5 file.
try:
    # First, try loading the model normally
    model = tf.keras.models.load_model("EthnoAgeNet.h5")
except:
    # If a ValueError or other error occurs, build the model from scratch
    # and load only the weights.
    model = create_model_architecture()
    model.load_weights("EthnoAgeNet.h5")

st.title("EthnoAgeNet - Age & Ethnicity Prediction")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Image preprocessing code from your previous message
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Read as a 3-channel color image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)

    # Predictions
    pred_age, pred_eth = model.predict(img_expanded)
    pred_age = int(pred_age[0][0])
    pred_eth = np.argmax(pred_eth)

    ethnicity_map = {0:"White", 1:"Black", 2:"Asian", 3:"Indian", 4:"Others"}

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Age:** {pred_age}")
    st.write(f"**Predicted Ethnicity:** {ethnicity_map[pred_eth]}")
