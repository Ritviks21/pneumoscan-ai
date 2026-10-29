
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Baseline CNN": "cnn_model.keras",
    "MobileNetV2 - Frozen": "mobilenet_frozen.keras",
    "MobileNetV2 - Fine-tuned": "mobilenet_finetuned.keras",
}


# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def prepare_image(image, model_name):
    image = image.convert("L").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)

    if model_name == "Baseline CNN":
        # The CNN applies its own 1/255 rescaling.
        array = array[np.newaxis, :, :, np.newaxis]
    else:
        # MobileNetV2 expects three channels and [-1, 1] input.
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        array = (array / 127.5) - 1.0
        array = array[np.newaxis, ...]

    return array


# --------------------------------------------------
# Dashboard UI
# --------------------------------------------------
st.title("PneumoScan AI")
st.subheader("Pediatric Chest X-ray Classification")

st.markdown(
    """
    Compare three trained deep-learning models on a chest X-ray image.
    This dashboard is for academic research and demonstration only.
    """
)

st.warning(
    "Research prototype only. Predictions are not medical diagnoses "
    "and must not guide clinical decisions."
)

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox(
        "Select a model",
        list(MODEL_FILES.keys()),
    )

    st.divider()
    st.caption("Input image size: 224 × 224 pixels")
    st.caption("Dataset: PneumoniaMNIST")
    st.caption("Models: CNN and MobileNetV2")


# --------------------------------------------------
# Upload and preview
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

        with right:
            st.markdown("### Selected model")
            st.info(model_name)
            st.markdown("### Image details")
            st.write(f"Image dimensions: {image.width} × {image.height}")

        if st.button("Analyze X-ray", type="primary"):
            model_path = BASE_DIR / MODEL_FILES[model_name]

            if not model_path.is_file():
                st.error(
                    f"Model file is missing: {MODEL_FILES[model_name]}"
                )
                st.stop()

            with st.spinner("Loading model and analyzing image..."):
                model = load_model(str(model_path))
                batch = prepare_image(image, model_name)
                output = np.asarray(
                    model.predict(batch, verbose=0)
                ).reshape(-1)

            if output.size == 0:
                st.error("The model returned no prediction.")
                st.stop()

            score = float(output[0])

            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                st.error("The model returned an invalid score.")
                st.stop()

            st.divider()
            st.header("Prediction result")

            # Assumes the training labels map sigmoid output >= 0.5
            # to the Pneumonia class.
            if score >= 0.5:
                st.error("Model prediction: Pneumonia")
            else:
                st.success("Model prediction: Normal class")

            st.metric("Pneumonia score", f"{score:.1%}")

            st.caption(
                "This score is the model's sigmoid output, not a "
                "clinically validated probability. The 0.5 threshold "
                "is used here for demonstration. Model predictions "
                "can be incorrect and must not guide medical care."
            )

    except Exception as exc:
        st.error("The image could not be analyzed.")
        st.exception(exc)

else:
    st.info("Upload a chest X-ray image to begin.")

st.divider()
st.caption(
    "PneumoScan AI | Academic project | "
    "CNN vs MobileNetV2 | Not for clinical use"
)
