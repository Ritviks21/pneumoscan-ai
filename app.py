
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "Baseline CNN": "cnn_model.keras",
    "MobileNetV2 - Frozen": "mobilenet_frozen.keras",
    "MobileNetV2 - Fine-tuned": "mobilenet_finetuned.keras",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path):
    # compile=False avoids loading the saved optimizer and training metrics.
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=True,
    )


def preprocess_image(image, model_name):
    image = image.convert("L").resize((224, 224))
    pixels = np.asarray(image, dtype=np.float32)

    if model_name == "Baseline CNN":
        # The baseline CNN rescales pixels internally.
        pixels = pixels[np.newaxis, :, :, np.newaxis]
    else:
        # Match the MobileNetV2 preprocessing used during training.
        pixels = np.repeat(pixels[..., np.newaxis], 3, axis=-1)
        pixels = pixels / 127.5 - 1.0
        pixels = pixels[np.newaxis, ...]

    return pixels


st.title("🫁 PneumoScan AI")
st.subheader("Pediatric Chest X-ray Classification")

st.write(
    "An academic demonstration comparing a baseline CNN "
    "with two MobileNetV2 transfer-learning models."
)

st.warning(
    "Research use only. This prototype is not a medical device. "
    "Predictions may be wrong and must not guide clinical decisions."
)

with st.sidebar:
    st.header("Model selection")
    selected_model = st.selectbox("Choose a model", list(MODELS))
    st.divider()
    st.write("Input size: 224 × 224")
    st.write("Dataset: PneumoniaMNIST")

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded X-ray", width=450)

        if st.button("Analyze X-ray", type="primary"):
            model_path = BASE_DIR / MODELS[selected_model]

            if not model_path.exists():
                st.error(f"Missing model file: {model_path.name}")
                st.stop()

            with st.spinner("Running model prediction..."):
                model = load_model(str(model_path))
                batch = preprocess_image(image, selected_model)
                prediction = np.asarray(
                    model.predict(batch, verbose=0)
                ).reshape(-1)

            if prediction.size != 1:
                st.error(
                    "Expected one sigmoid output, but the model "
                    "returned a different output shape."
                )
                st.stop()

            score = float(prediction[0])

            if not np.isfinite(score) or not 0 <= score <= 1:
                st.error("The model returned an invalid score.")
                st.stop()

            st.divider()
            st.subheader("Model output")
            st.write(f"Selected model: **{selected_model}**")
            st.metric("Sigmoid output", f"{score:.4f}")

            st.caption(
                "The score is the raw sigmoid output. Confirm the "
                "training label mapping before interpreting it as "
                "Pneumonia probability or assigning a class label."
            )

    except Exception as exc:
    st.error("Model loading or prediction failed.")
    st.exception(exc)

else:
    st.info("Upload a chest X-ray to begin.")

st.divider()
st.caption("PneumoScan AI | Student research prototype")
