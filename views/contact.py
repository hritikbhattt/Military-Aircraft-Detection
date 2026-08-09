import streamlit as st
from theme import inject_css, top_nav, page_header, MAIL_ICON

inject_css()
top_nav("Contact")
page_header(MAIL_ICON, "Contact & Links", "Get in touch or explore more")

st.markdown("""
<div class="contact-grid">
    <a class="contact-card" href="mailto:Hritikbhatt18@gmail.com">
        <div class="contact-icon">📧</div>
        <div class="contact-label">Email</div>
        <div class="contact-value">Hritikbhatt18@gmail.com</div>
    </a>
    <a class="contact-card" href="https://linkedin.com/in/hritikbhatt" target="_blank">
        <div class="contact-icon">💼</div>
        <div class="contact-label">LinkedIn</div>
        <div class="contact-value">linkedin.com/in/hritikbhatt</div>
    </a>
    <a class="contact-card" href="https://github.com/hritikbhattt" target="_blank">
        <div class="contact-icon">💻</div>
        <div class="contact-label">GitHub</div>
        <div class="contact-value">github.com/hritikbhattt</div>
    </a>
</div>

<div class="panel">
<h3>More Projects</h3>
<p>🔗 <a href="https://github.com/hritikbhattt/Military-Aircraft-Detection" target="_blank">
Military Aircraft Detection — full source code, README, and per-class results</a></p>
<p>🔗 <a href="https://github.com/hritikbhattt/Business-intelligence-dashboard" target="_blank">
Business Intelligence Dashboard — Superstore sales analysis (Zoho Analytics + Excel)</a></p>
</div>
""", unsafe_allow_html=True)
