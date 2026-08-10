import streamlit as st
import os
from theme import inject_css, top_nav, page_header, CHART_ICON

inject_css()
top_nav("Results")
page_header(CHART_ICON, "Results", "Evaluated on the held-out test split (1,572 images)")

st.markdown("""
<div class="hero-wrap">
<div class="bar-row">
    <div class="bar-label"><span>mAP50</span><span>65.5%</span></div>
    <div class="bar-track"><div class="bar-fill" style="--target-width:65.5%;"></div></div>
</div>
<div class="bar-row">
    <div class="bar-label"><span>mAP50-95</span><span>57.8%</span></div>
    <div class="bar-track"><div class="bar-fill" style="--target-width:57.8%;"></div></div>
</div>
<div class="bar-row">
    <div class="bar-label"><span>Precision</span><span>70.3%</span></div>
    <div class="bar-track"><div class="bar-fill" style="--target-width:70.3%;"></div></div>
</div>
<div class="bar-row">
    <div class="bar-label"><span>Recall</span><span>55.9%</span></div>
    <div class="bar-track"><div class="bar-fill" style="--target-width:55.9%;"></div></div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
<h3>Summary</h3>
<p>Trained for 40 epochs on the full 17,687-image training set (YOLOv8s, 640px, batch 16), then
evaluated on 1,572 test images never seen during training or checkpoint selection. Performance is
strong on well-represented classes (<code>X32</code>: 0.96 mAP50, <code>C2</code>: 0.92,
<code>US2</code>: 0.93) and noticeably weaker on classes with very few training examples
(<code>B21</code>: 3 test images, <code>MQ28</code>: 3 test images) — a direct, expected consequence
of class imbalance in the source dataset rather than a modeling flaw.</p>
</div>
""", unsafe_allow_html=True)

chart_path = "outputs/eval/per_class_ap.png"
st.markdown('<div class="panel"><h3>Per-Class AP Breakdown</h3>', unsafe_allow_html=True)
if os.path.exists(chart_path):
    st.image(chart_path, width='stretch')
else:
    st.info("Per-class chart not found in this deployment. See the full breakdown on GitHub.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="panel">
<h3>Known Issues <span class="badge-warn">Transparent by design</span></h3>
<p>The project's custom metrics module (an independent cross-check against Ultralytics' own validator)
reports a somewhat lower mAP50 due to differences in confidence-threshold filtering and AP-interpolation
convention — both are valid but non-identical approaches. The Ultralytics-reported numbers above are the
primary, trusted metrics for this project.</p>
<p>Full technical detail is documented in the
<a href="https://github.com/hritikbhattt/Military-Aircraft-Detection#known-issues" target="_blank">
GitHub README's Known Issues section</a>.</p>
</div>
""", unsafe_allow_html=True)
