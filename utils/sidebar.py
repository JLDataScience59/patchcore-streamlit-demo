# utils/sidebar.py
import streamlit as st
from pathlib import Path

def render_sidebar():
    static_dir = Path(__file__).parent.parent / "static"
    with st.sidebar:
        st.image(str(static_dir / "datascientest_logo.png"), width=100)
        st.markdown("**[DataScientest](https://datascientest.com/)**")
        st.markdown("Projet DS - Promotion Bootcamp Mai 2025") 
        st.markdown("---")
        
        st.markdown("### 👤 Equipe projet")
        st.image(str(static_dir / "jeremy.jpg"), width=100)
        st.markdown("**Jérémy LESOT [LinkedIn](https://linkedin.com/in/jeremy-lesot)**")

        st.image(str(static_dir / "eric.jpg"), width=100)
        st.markdown("**Eric MATHIEU [Linkedin](https://www.linkedin.com/in/ericmathieu2/)**")

        st.image(str(static_dir / "alexandre.jpg"), width=100)
        st.markdown("**Alexandre MATHIEU [Linkedin](https://www.linkedin.com/in/alexandre-g-mathieu/)**")

        st.image(str(static_dir / "roberto.png"), width=100)
        st.markdown("**Roberto Vercellin [Linkedin](https://www.linkedin.com/in/roberto-vercellin-3b4a18108/)**")