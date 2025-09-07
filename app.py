import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the custom model architecture to match the saved H5 file
# This is a critical step to ensure the deserializer can correctly map weights
def create_model_architecture():
    # The input shape is hard-coded to 128x128x3 as suggested by the error.
    inputs = Input(shape=(128, 128, 3)) 
    
    # Define the layers from your original model
    # The error is in the 'stem_conv' layer, so define it correctly.
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='stem_conv')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4, name='dropout_1')(x) # Your problematic layer
    # ...add the rest of your model's layers here...
    
    # Placeholder for the rest of your model's architecture
    # Replace these with your actual layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    # Assuming two outputs: age and ethnicity
    age_output = Dense(1, activation='relu', name='age_output')(x)
    ethnicity_output = Dense(5, activation='softmax', name='ethnicity_output')(x)

    model = Model(inputs=inputs, outputs=[age_output, ethnicity_output])
    
    return model

# Load the model with custom objects to handle deserialization
try:
    model = tf.keras.models.load_model("EthnoAgeNet.h5", custom_objects={
        'Conv2D': Conv2D,
        'Dropout': Dropout,
        'BatchNormalization': BatchNormalization,
        # Add any other custom layers you may have
    })
    
except ValueError:
    # If loading with custom objects fails, create the architecture and load weights manually
    st.write("Shape mismatch detected. Attempting manual weight loading...")
    
    # Build the model architecture from scratch
    model = create_model_architecture()
    
    # Load the weights into the new model
    model.load_weights("EthnoAgeNet.h5")
    
    st.write("Model loaded successfully via manual weight loading.")

st.title("EthnoAgeNet - Age & Ethnicity Prediction")

# Rest of your Streamlit app code
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
    
    img_expanded = np.expand_dims(img_resized, axis=0)

    pred_age, pred_eth = model.predict(img_expanded)
    pred_age = int(pred_age[0][0])
    pred_eth = np.argmax(pred_eth)

    ethnicity_map = {0:"White",1:"Black",2:"Asian",3:"Indian",4:"Others"}

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Age:** {pred_age}")
    st.write(f"**Predicted Ethnicity:** {ethnicity_map[pred_eth]}")
