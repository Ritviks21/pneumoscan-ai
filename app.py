
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Baseline CNN": "cnn_model.keras",
    "MobileNetV2 — Frozen": "mobilenet_frozen.keras",
    "MobileNetV2 — Fine-tuned": "mobilenet_finetuned.keras",
}


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def prepare_image(image, model_name):
    image = image.convert("L").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)

    if model_name == "Baseline CNN":
        # CNN uses raw pixel values; its model rescales internally.
        array = array[np.newaxis, :, :, np.newaxis]
    else:
        # MobileNetV2 was trained with 3 channels and [-1, 1] pixels.
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        array = (array / 127.5) - 1.0
        array = array[np.newaxis, ...]

    return array


st.title("🫁 PneumoScan AI")
st.subheader("Pediatric Chest X-ray Classification")

st.write(
    "Compare a baseline CNN with two MobileNetV2 transfer-learning models."
)

st.info(
    "Research prototype only. Predictions are not medical diagnoses "
    and must not guide clinical decisions."
)

model_name = st.selectbox(
    "Choose a trained model",
    list(MODEL_FILES.keys()),
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyze X-ray", type="primary"):
        model_path = BASE_DIR / MODEL_FILES[model_name]

        if not model_path.is_file():
            st.error(f"Model file not found: {model_path.name}")
            st.stop()

        try:
            with st.spinner("Loading model and analyzing image..."):
                model = load_model(str(model_path))
                batch = prepare_image(image, model_name)
                prediction = np.asarray(
                    model.predict(batch, verbose=0)
                ).reshape(-1)

            if prediction.size == 0:
                st.error("The model returned no prediction.")
                st.stop()

            score = float(prediction[0])

            if not np.isfinite(score) or not 0 <= score <= 1:
                st.error("The model returned an invalid probability.")
                st.stop()

            st.subheader("Prediction result")

            if score >= 0.5:
                st.error("Model prediction: Pneumonia")
                st.metric("Pneumonia score", f"{score:.1%}")
            else:
                st.success("Model prediction: Normal class")
                st.metric("Pneumonia score", f"{score:.1%}")

            st.caption(
                "The score is the model's sigmoid output, not a "
                "validated clinical probability. The 0.5 threshold "
                "is a demonstration threshold."
            )

        except Exception as exc:
            st.error("Analysis failed. Check the deployment logs.")
            st.exception(exc)

st.divider()
st.caption(
    "Academic project | CNN vs MobileNetV2 | "
    "Not intended for clinical use"
)
