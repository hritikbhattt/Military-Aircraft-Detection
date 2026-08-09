import streamlit as st
from theme import inject_css, top_nav, page_header, INFO_ICON

inject_css()
top_nav("About")
page_header(INFO_ICON, "About the Project", "Design decisions, dataset, and pipeline")

st.markdown("""
<div class="panel">
<h3>Overview</h3>
<p>This project trains a YOLOv8 object detector to localize and classify military aircraft in still
images and video, across a full 103-class taxonomy. It covers the complete applied machine learning
lifecycle — not just a trained model file, but dataset preparation, configurable training, evaluation
against a held-out test set, and a usable inference pipeline for images, video, and webcam input.</p>
</div>

<div class="panel">
<h3>Tech Stack</h3>
<span class="tag">Python</span>
<span class="tag">YOLOv8 (Ultralytics)</span>
<span class="tag">PyTorch</span>
<span class="tag">OpenCV</span>
<span class="tag">Streamlit</span>
<span class="tag">GitHub Actions CI</span>
<span class="tag">Kaggle (GPU training)</span>
</div>

<div class="panel">
<h3>Dataset</h3>
<p>Trained on the <a href="https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset" target="_blank">
Military Aircraft Detection Dataset</a> (Kaggle) — one row per bounding box across 103 aircraft classes,
with a pre-assigned train/validation/test split. Full-dataset split sizes: <b>17,687 train / 4,361
validation / 1,572 test</b> images.</p>
<p>Several classes are naturally rare in the source data (e.g. <code>WZ9</code>: 13 images,
<code>MQ20</code>: 16 images) and perform noticeably worse as a direct, expected consequence of class
imbalance — this is disclosed honestly rather than hidden. See the Results page for the full breakdown.</p>
</div>

<div class="panel">
<h3>Design Decisions</h3>
<p><b>103 vs. 36 classes:</b> an earlier scope draft targeted a 36-class subset. The decision was made
to use the full 103-class taxonomy present in the source dataset instead, since the additional classes
were already fully labeled and available at no extra cost.</p>
<p><b>Independent metrics module:</b> alongside Ultralytics' own validator, the project includes a
from-scratch, pure-numpy IoU/AP implementation as a correctness cross-check against Ultralytics'
internal box-matching conventions.</p>
<p><b>Testing & CI:</b> 19 unit tests cover the box-conversion math and metrics implementation,
enforced on every push via GitHub Actions.</p>
</div>

<div class="panel">
<h3>Supported Aircraft (103 classes)</h3>
<p>Search to filter the full list of aircraft this model can recognize.</p>
</div>
""", unsafe_allow_html=True)

ALL_CLASSES = sorted([
    "A10", "A400M", "AG600", "AH64", "AKINCI", "AV8B", "An124", "An22", "An225", "An72",
    "B1", "B2", "B21", "B52", "Be200", "C1", "C130", "C17", "C2", "C390", "C5", "CH47",
    "CH53", "CL415", "E2", "E7", "EF2000", "EMB314", "F117", "F14", "F15", "F16", "F18",
    "F2", "F22", "F35", "F4", "FCK1", "H6", "Il76", "J10", "J20", "J35", "J36", "J50",
    "JAS39", "JF17", "JH7", "KAAN", "KC135", "KF21", "KIZILELMA", "KJ600", "Ka27", "Ka52",
    "MQ20", "MQ25", "MQ28", "MQ35", "MQ9", "Mi24", "Mi26", "Mi28", "Mi8", "Mig29", "Mig31",
    "Mirage2000", "NH90", "P3", "RQ4", "Rafale", "SR71", "Su24", "Su25", "Su34", "Su47",
    "Su57", "T50", "TB001", "TB2", "Tejas", "Tornado", "Tu160", "Tu22M", "Tu95", "U2",
    "UH60", "US2", "V22", "V280", "Vulcan", "WZ10", "WZ7", "WZ9", "X29", "X32", "XB70",
    "XQ58", "Y20", "YF23", "Z10", "Z19", "Z21",
])

search = st.text_input("🔍 Search aircraft classes", placeholder="e.g. F16, Rafale, Su57...")
filtered = [c for c in ALL_CLASSES if search.lower() in c.lower()] if search else ALL_CLASSES

st.markdown('<div class="panel">', unsafe_allow_html=True)
if not filtered:
    st.write("No matching aircraft found.")
else:
    st.caption(f"Showing {len(filtered)} of {len(ALL_CLASSES)} classes")
    chips = "".join(f'<span class="tag">{c}</span>' for c in filtered)
    st.markdown(chips, unsafe_allow_html=True)
st.markdown("""
</div>

<div class="panel">
<h3>Pipeline</h3>
<p>Raw dataset (<code>labels_with_split.csv</code> + images) → <code>data/prepare_dataset.py</code>
(VOC → YOLO box conversion) → <code>data/images/&#123;train,validation,test&#125;</code> →
<code>scripts/train.py</code> (YOLOv8 fine-tuning) → <code>models/best.pt</code> → used by both
<code>scripts/evaluate.py</code> (per-class AP, mAP, precision/recall) and
<code>scripts/detect.py</code> (image / video / webcam inference).</p>
</div>
""", unsafe_allow_html=True)
