
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# --------------------------------------------------
# Page setup
# --------------------------------------------------
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

# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path.name}"
        )

    # compile=False avoids restoring training optimizer state.
    return tf.keras.models.load_model(
        str(path),
        compile=False,
    )


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def prepare_image(image, model_name, model):
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((224, 224))

    pixels = np.asarray(image, dtype=np.float32)

    # Check the saved model's expected input channels.
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError(
            "This app expects a model with one image input."
        )

    channels = input_shape[-1]

    if channels == 1:
        # Baseline CNN applies its own 1/255 rescaling.
        batch = pixels[np.newaxis, :, :, np.newaxis]

    elif channels == 3:
        # MobileNetV2 training used grayscale repeated to RGB
        # and pixel values scaled from [0, 255] to [-1, 1].
        rgb = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)
        rgb = rgb / 127.5 - 1.0
        batch = rgb[np.newaxis, ...]

    else:
        raise ValueError(
            f"Unsupported model input shape: {input_shape}"
        )

    return batch


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
st.title("🫁 PneumoScan AI")
st.subheader("Pediatric Chest X-ray Classification")

st.write(
    "Compare a baseline CNN with frozen and fine-tuned "
    "MobileNetV2 models using the PneumoniaMNIST benchmark."
)

st.warning(
    "Academic research prototype only. This application is "
    "not a medical device and must not be used for diagnosis "
    "or treatment decisions."
)

with st.sidebar:
    st.header("Model selection")

    selected_model = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
    )

    st.divider()
    st.write("Input size: 224 × 224")
    st.write("Dataset: PneumoniaMNIST")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload a chest X-ray to begin.")

else:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

        with right:
            st.markdown("### Selected model")
            st.write(selected_model)
            st.markdown("### Image dimensions")
            st.write(f"{image.width} × {image.height} pixels")

        if st.button("Analyze X-ray", type="primary"):
            model_path = BASE_DIR / MODELS[selected_model]

            try:
                with st.spinner(
                    f"Loading {selected_model} and predicting..."
                ):
                    model = load_model(str(model_path))
                    batch = prepare_image(
                        image,
                        selected_model,
                        model,
                    )

                    output = model.predict(batch, verbose=0)
                    output = np.asarray(output)

                if output.size != 1:
                    st.error(
                        "Expected one sigmoid output, but the model "
                        f"returned shape {output.shape}."
                    )
                    st.stop()

                score = float(output.reshape(-1)[0])

                if not np.isfinite(score) or not 0 <= score <= 1:
                    st.error(
                        "The model output is not a valid sigmoid score."
                    )
                    st.stop()

                st.divider()
                st.subheader("Model output")

                st.write(f"**Selected model:** {selected_model}")
                st.metric("Raw sigmoid output", f"{score:.4f}")

                st.caption(
                    "This is the model's raw output, not a verified "
                    "clinical probability. Confirm the training label "
                    "mapping before interpreting the score as a class."
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed for {selected_model}."
                )
                st.code(
                    f"{type(exc).__name__}: {exc}",
                    language="text",
                )
                st.info(
                    "Share this error if the model still fails. "
                    "It identifies the actual loading or prediction issue."
                )

    except Exception as exc:
        st.error("The uploaded image could not be opened.")
        st.code(
            f"{type(exc).__name__}: {exc}",
            language="text",
        )

st.divider()
st.caption("PneumoScan AI | Student research prototype")
