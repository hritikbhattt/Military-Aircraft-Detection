import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Military Aircraft Detection", page_icon="✈️", layout="centered")

MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.title("✈️ Military Aircraft Detection")
st.markdown(
    "YOLOv8 model trained to detect and classify **103 military aircraft types** "
    "(fighters, bombers, transports, helicopters, UAVs). "
    "Achieves **65.5% mAP50** on a held-out test set of 1,572 images. "
    "[View the full project on GitHub](https://github.com/hritikbhattt/Military-Aircraft-Detection)."
)

conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this demo**\n\n"
    "Upload a photo, or take one with your camera, and the model will draw bounding "
    "boxes around any aircraft it recognizes, with the predicted class and confidence score."
)

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

input_image = None

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with tab2:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert("RGB")

if input_image is not None:
    st.markdown("### Original")
    st.image(input_image, use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(source=np.array(input_image), conf=conf_threshold, verbose=False)
        result = results[0]
        annotated = result.plot()  # returns BGR numpy array

    st.markdown("### Detection Result")
    st.image(annotated, channels="BGR", use_container_width=True)

    num_detections = len(result.boxes)
    if num_detections == 0:
        st.info("No aircraft detected above the current confidence threshold. Try lowering it in the sidebar.")
    else:
        st.markdown(f"### Detected {num_detections} aircraft")
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"**{cls_name}** — confidence: {conf:.2f}")
else:
    st.info("Upload an image or take a photo above to get started.")

st.markdown("---")
st.caption(
    "Model trained on the Military Aircraft Detection Dataset (Kaggle) using YOLOv8s, "
    "40 epochs, 17,687 training images. Some rare classes (fewer than 20 training images) "
    "perform noticeably worse — see the GitHub README for full per-class results."
)
