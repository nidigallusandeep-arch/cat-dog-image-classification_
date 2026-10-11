"""Streamlit interface for the trained Cat vs Dog classifier."""
from pathlib import Path
import json
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path("artifacts/cat_dog_model.keras")
META_PATH = Path("artifacts/metadata.json")
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱", layout="centered")
st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload a photo and the trained model will predict whether it looks like a cat or a dog.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run `python train_model.py` and make sure artifacts/cat_dog_model.keras exists.")
    st.stop()

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = ["cats", "dogs"]
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text())
    class_names = metadata.get("class_names", class_names)
    IMG_SIZE = tuple(metadata.get("image_size", list(IMG_SIZE)))

uploaded_file = st.file_uploader("Choose a cat or dog image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        resized = image.resize(IMG_SIZE)
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        score = float(model.predict(batch, verbose=0)[0][0])

        # Keras directory labels are alphabetical; binary label 0 is class_names[0],
        # and label 1 is class_names[1]. Sigmoid output estimates class 1.
        predicted_index = 1 if score >= 0.5 else 0
        confidence = score if predicted_index == 1 else 1.0 - score
        st.subheader(f"Prediction: {class_names[predicted_index].title()}")
        st.progress(int(round(confidence * 100)))
        st.write(f"Model score for **{class_names[1]}**: {score:.1%}")
        st.caption(f"Confidence for predicted class: {confidence:.1%}. This is a model score, not a guarantee.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image to try the model.")

st.caption("Educational demo. Performance depends on dataset quality and training; test on images not used during training.")
