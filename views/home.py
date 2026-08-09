import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import glob
import os
from theme import inject_css, top_nav

inject_css()
top_nav("Home")

MODEL_PATH = "models/best.pt"
SAMPLES_DIR = "test_files"
CONFIDENCE = 0.25


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.markdown("""
<div class="desc-card">
    A YOLOv8 object detector trained to identify and classify <b>103 military aircraft types</b> —
    fighters, bombers, transports, helicopters, and UAVs. Upload a photo, use your camera, or try one
    of the sample images below — the model will draw bounding boxes around every aircraft it recognizes,
    with its predicted class and confidence score.
    &nbsp;<a href="https://github.com/hritikbhattt/Military-Aircraft-Detection" target="_blank">View full project on GitHub →</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-card"><div class="lbl">Aircraft Classes</div><div class="num accent">103</div></div>
    <div class="stat-card"><div class="lbl">mAP50 (Test Set)</div><div class="num accent">65.5%</div></div>
    <div class="stat-card"><div class="lbl">Training Images</div><div class="num">17,687</div></div>
    <div class="stat-card"><div class="lbl">Test Images</div><div class="num">1,572</div></div>
</div>
""", unsafe_allow_html=True)

if "sample_choice" not in st.session_state:
    st.session_state.sample_choice = None

tab1, tab2, tab3 = st.tabs(["📁  Upload Image", "📷  Camera", "🖼️  Try a Sample"])

input_image = None

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.sample_choice = None
        input_image = Image.open(uploaded_file).convert("RGB")

with tab2:
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file is not None:
        st.session_state.sample_choice = None
        input_image = Image.open(camera_file).convert("RGB")

with tab3:
    sample_paths = sorted(
        p for p in glob.glob(os.path.join(SAMPLES_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png")) and "_out" not in os.path.basename(p)
    )
    if not sample_paths:
        st.info("No sample images found in test_files/.")
    else:
        cols = st.columns(min(len(sample_paths), 5))
        for i, path in enumerate(sample_paths):
            with cols[i % len(cols)]:
                st.image(path, width='stretch')
                label = os.path.splitext(os.path.basename(path))[0]
                if st.button(f"Use this ({label})", key=f"sample_btn_{i}"):
                    st.session_state.sample_choice = path

if st.session_state.sample_choice:
    input_image = Image.open(st.session_state.sample_choice).convert("RGB")

if input_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel"><h4>Original</h4>', unsafe_allow_html=True)
        st.image(input_image, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Running detection..."):
        results = model.predict(source=np.array(input_image), conf=CONFIDENCE, verbose=False)
        result = results[0]
        annotated = result.plot()

    with col2:
        st.markdown('<div class="panel"><h4>Detection Result</h4>', unsafe_allow_html=True)
        st.image(annotated, channels="BGR", width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    num_detections = len(result.boxes)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if num_detections == 0:
        st.info("No aircraft detected above the confidence threshold.")
    else:
        st.markdown(f"<h4>Detected {num_detections} aircraft</h4>", unsafe_allow_html=True)
        chips_html = ""
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            chips_html += f'<span class="detection-chip"><b>{cls_name}</b> · {conf:.2f}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload an image, take a photo, or try a sample above to get started.")

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Model trained on the Military Aircraft Detection Dataset (Kaggle) using YOLOv8s, "
    "40 epochs, 17,687 training images."
)
