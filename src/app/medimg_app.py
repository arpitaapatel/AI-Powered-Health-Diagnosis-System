import os
import io
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from medimg.data import _normalize
from medimg.utils import compute_gradcam_heatmap, overlay_heatmap_on_image


st.set_page_config(page_title="Chest X-ray Diagnosis", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image, image_size=(224, 224)) -> tf.Tensor:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(image_size)
    arr = np.array(img)
    tensor = tf.convert_to_tensor(arr)[None, ...]
    tensor = _normalize(tensor)
    return tensor


def load_class_names_for_model(model_path: str, num_classes: int) -> list:
    base, _ = os.path.splitext(model_path)
    sidecar = f"{base}_class_names.json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                names = json.load(f)
                if isinstance(names, list) and len(names) == num_classes:
                    return names
        except Exception:
            pass
    # Fallback generic labels
    if num_classes == 3:
        return ["COVID-19", "Normal", "Pneumonia"]
    return [f"Class {i}" for i in range(num_classes)]


def main():
    st.title("Medical Imaging AI: Chest X-ray Classification")
    st.write("Upload a chest X-ray to classify as Normal / Pneumonia / COVID-19 and visualize Grad-CAM.")

    with st.sidebar:
        st.header("Model Selection")
        model_choice = st.selectbox("Choose model", ["efficientnetb0_finetuned.keras", "resnet50_finetuned.keras", "resnet50.keras", "efficientnetb0.keras", "scratch_cnn.keras"]) 
        model_dir = st.text_input("Model directory", value="artifacts")
        image_size = st.select_slider("Image size", options=[224, 256, 299], value=224)

    uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input X-ray")
            st.image(img, use_column_width=True)

        model_path = os.path.join(model_dir, model_choice)
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Train models first.")
            return
        model = load_model(model_path)

        x = preprocess_image(img, image_size=(image_size, image_size))
        preds = model.predict(x, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        class_names = load_class_names_for_model(model_path, preds.shape[1])

        pred_label = class_names[pred_idx]

        with col2:
            st.subheader("Prediction")
            st.metric("Predicted Class", pred_label, delta=f"Confidence: {confidence:.2%}")

        try:
            heatmap = compute_gradcam_heatmap(model, x, class_index=pred_idx)
            overlay = overlay_heatmap_on_image(np.array(img.convert('RGB')), heatmap, alpha=0.4)
            st.subheader("Grad-CAM Heatmap")
            st.image(overlay, use_column_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")


if __name__ == "__main__":
    main()


