import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# ── Page setup ───────────────────────
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="centered"
)

st.title("🥔 Potato Disease Detector")
st.write("Upload a potato leaf image and get prediction instantly")

# ── Model download from Google Drive ─
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1lEsUcnbuwV2tGraxq4OCQ_82Jdx1vmTt" # 👉 এখানে তোমার direct download link দিবা
    gdown.download(url, MODEL_PATH, quiet=False)

# ── Load model ───────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Class names ───────────────────────
class_names = ['Early_blight', 'Late_blight', 'Healthy']

# ── Upload image ──────────────────────
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100

    # result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

else:
    st.info("Please upload a leaf image to start prediction.")
