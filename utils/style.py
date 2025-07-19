# utils/style.py

import streamlit as st
from pathlib import Path

def apply_custom_css():
    css_path = Path(__file__).parent.parent / "style" / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
