import streamlit as st
from pathlib import Path

from utils.style import apply_custom_css
from utils.sidebar import render_sidebar

apply_custom_css()
render_sidebar()

static_dir = Path(__file__).parent.parent / "static"

st.title("📂 Jeu de données : MVTec AD")

st.markdown("""
Le jeu de données **MVTec Anomaly Detection (MVTec AD)** est un standard pour la détection d’anomalies visuelles sur des pièces industrielles.

Il contient plusieurs catégories d’objets et textures, chacun avec :
- Des images normales (good)
- Des images défectueuses (defective, plusieurs types de défauts)
- Des masques de ground-truth pour les anomalies

🔗 [Site officiel MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
""")


st.markdown("### 📁 Structure du dataset (exemple : bottle)")
st.code("""
bottle/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── broken_small/
│   └── broken_large/
├── ground_truth/
    ├── broken_small/
    └── broken_large/
""", language="bash")

st.markdown("### 📸 Exemples d’images")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(str(static_dir / "bottle_good.png"), caption="Exemple : good", width=220)

with col2:
    st.image(str(static_dir / "bottle_defective.png"), caption="Exemple : defective", width=220)

with col3:
    st.image(str(static_dir / "bottle_mask.png"), caption="Exemple : ground_truth", width=220)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(str(static_dir / "screw_good.png"), caption="Exemple : good", width=220)

with col2:
    st.image(str(static_dir / "screw_defective.png"), caption="Exemple : defective", width=220)

with col3:
    st.image(str(static_dir / "screw_mask.png"), caption="Exemple : ground_truth", width=220)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(str(static_dir / "hazelnut_good.png"), caption="Exemple : good", width=220)

with col2:
    st.image(str(static_dir / "hazelnut_defective.png"), caption="Exemple : defective", width=220)

with col3:
    st.image(str(static_dir / "hazelnut_mask.png"), caption="Exemple : ground_truth", width=220)