import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

# -------------------
# Build model (must match training)
# -------------------
def build_model():
    # Use weights=None to avoid downloading weights from a URL
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(128, 128, 3))
    x = base_model.output

    # The original model weights were saved using a Flatten layer, not GlobalAveragePooling2D.
    # We must use Flatten to match the architecture of the saved weights.
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Age head
    age_output = Dense(1, activation='linear', name='age_output')(x)

    # Ethnicity head
    ethnicity_output = Dense(5, activation='softmax', name='ethnicity_output')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, ethnicity_output])
    return model

# -------------------
# Load weights (not the whole model)
# -------------------
model = build_model()
try:
    model.load_weights("EthnoAgeNet.h5")
    st.success("Model weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# -------------------
# Streamlit UI
# -------------------
st.title("EthnoAgeNet - Age & Ethnicity Prediction")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Preprocess
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predict
    pred_age, pred_ethnicity = model.predict(img_input)

    # Results
    age = int(pred_age[0][0])
    ethnicity_idx = np.argmax(pred_ethnicity[0])
    ethnicity_labels = ["White", "Black", "Asian", "Indian", "Others"]
    ethnicity = ethnicity_labels[ethnicity_idx]

    # Show
    st.image(img_rgb, caption=f"Predicted Age: {age}, Ethnicity: {ethnicity}", use_column_width=True)
    st.write(f"### Predicted Age: {age}")
    st.write(f"### Predicted Ethnicity: {ethnicity}")
