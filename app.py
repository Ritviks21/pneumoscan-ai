
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------
# Page configuration
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

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Load one model when selected
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
# Preprocess image to match the saved model
# --------------------------------------------------
def prepare_image(image, model):
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize(IMAGE_SIZE)

    pixels = np.asarray(image, dtype=np.float32)
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError(
            "This application supports models with one image input."
        )

    if len(input_shape) != 4:
        raise ValueError(
            f"Expected a 4D image input, got {input_shape}."
        )

    height, width, channels = input_shape[1:]

    if height not in (None, IMAGE_SIZE[1]) or width not in (
        None,
        IMAGE_SIZE[0],
    ):
        raise ValueError(
            f"Model expects input shape {input_shape}, "
            "but this app preprocesses images to 224 × 224."
        )

    if channels == 1:
        # Baseline CNN rescales [0, 255] internally.
        batch = pixels[np.newaxis, :, :, np.newaxis]

    elif channels == 3:
        # Match the MobileNetV2 training preprocessing.
        rgb = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)
        rgb = rgb / 127.5 - 1.0
        batch = rgb[np.newaxis, ...]

    else:
        raise ValueError(
            f"Unsupported model input channels: {channels}. "
            f"Model input shape: {input_shape}"
        )

    return batch


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
st.title("🫁 PneumoScan AI")
st.subheader("Pediatric Chest X-ray Classification")

st.write(
    "Compare a baseline CNN with frozen and fine-tuned "
    "MobileNetV2 models on the PneumoniaMNIST benchmark."
)

st.warning(
    "Research prototype only. This application is not a medical "
    "device. Do not use its predictions to diagnose or treat anyone."
)

with st.sidebar:
    st.header("Model selection")

    selected_model = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
    )

    st.divider()
    st.write("Input size: 224 × 224 pixels")
    st.write("Dataset: PneumoniaMNIST")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload a chest X-ray image to begin.")

else:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded X-ray",
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
                    f"Loading {selected_model} and analyzing image..."
                ):
                    model = load_model(str(model_path))
                    batch = prepare_image(image, model)

                    output = model.predict(batch, verbose=0)
                    output = np.asarray(output)

                if output.size != 1:
                    raise ValueError(
                        "Expected exactly one sigmoid output, "
                        f"but the model returned shape {output.shape}."
                    )

                score = float(output.reshape(-1)[0])

                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"Expected a sigmoid score from 0 to 1; "
                        f"received {score}."
                    )

                # Assumes training labels are 0 = Normal,
                # 1 = Pneumonia and the sigmoid output represents class 1.
                pneumonia_score = score

                st.divider()
                st.subheader("Prediction result")
                st.write(f"**Model:** {selected_model}")

                if pneumonia_score >= 0.5:
                    st.error("YES — Model predicts Pneumonia")
                    displayed_score = pneumonia_score
                    score_label = "Pneumonia model score"
                else:
                    st.success("NO — Model predicts Normal")
                    displayed_score = 1.0 - pneumonia_score
                    score_label = "Normal model score"

                st.metric(
                    score_label,
                    f"{displayed_score * 100:.1f}%",
                )

                st.progress(displayed_score)

                st.caption(
                    "The percentage is a model score derived from its "
                    "sigmoid output, not a medically verified probability. "
                    "The YES/NO mapping assumes class 0 = Normal and "
                    "class 1 = Pneumonia. Predictions can be incorrect."
                )

            except Exception as exc:
                st.error(
                    f"Analysis failed for {selected_model}. "
                    "The details are shown below."
                )
                st.code(
                    f"{type(exc).__name__}: {exc}",
                    language="text",
                )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error("The uploaded image could not be opened.")
        st.code(
            f"{type(exc).__name__}: {exc}",
            language="text",
        )

st.divider()
st.caption("PneumoScan AI | Academic research prototype")
