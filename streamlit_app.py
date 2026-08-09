import streamlit as st

st.set_page_config(page_title="Military Aircraft Detection", page_icon="✈️", layout="wide")

pg = st.navigation(
    [
        st.Page("views/home.py", title="Home", url_path="", default=True),
        st.Page("views/about.py", title="About", url_path="About"),
        st.Page("views/results.py", title="Results", url_path="Results"),
        st.Page("views/contact.py", title="Contact", url_path="Contact"),
    ],
    position="hidden",
)
pg.run()
