import streamlit as st
from utils.style import apply_custom_css
from utils.sidebar import render_sidebar
from pathlib import Path

apply_custom_css()
render_sidebar()

# CSS personnalisé pour élargir encore plus le contenu
st.markdown("""
    <style>
        .main .block-container {
            max-width: 95%;
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])  # Ratio : 25% / 50% / 25%

with col2:
    image_path = Path(__file__).parent.parent / "static" / "defectAI.png"
    st.image(str(image_path), use_container_width=True)

st.write("# Détection d'anomalies dans des pièces industrielles")

st.markdown(
    """
    Dans le cadre d’un parcours de formation intensive de trois mois en data science, dispensé par l'organisme d’e-learning [DataScientest](https://datascientest.com/), notre groupe de quatre apprenants a mené un projet appliqué visant à développer une solution de détection automatique d’anomalies sur des pièces industrielles à partir d’images.

    Ce projet s’inscrit dans une problématique industrielle concrète, où la qualité des pièces produites est un enjeu critique. L’objectif principal est de concevoir un modèle de vision par ordinateur capable de détecter automatiquement la présence d’anomalies visuelles, telles que des fissures, rayures, ou défauts de fabrication.

    Nous explorons différentes approches de machine learning, notamment des architectures de réseaux de neurones convolutifs (CNN), en supervision ou non supervision, selon les contraintes de disponibilité des données annotées. Deux axes principaux structurent notre démarche :

    **1. Une classification binaire** : déterminer si une pièce est conforme (good) ou présente une anomalie (defective).

    **2. Une classification multi-classes** : identifier précisément le type de défaut parmi ceux répertoriés pour chaque catégorie d’objet.

    Ce travail s’appuie sur le jeu de données ouvert MVTec AD, qui propose des images de pièces industrielles issues de différentes catégories, annotées pour l'entraînement et la validation de modèles de détection.

    Afin de contextualiser et valoriser notre démarche d’un point de vue métier, nous avons imaginé un cas d’usage fictif, représenté par l’entreprise TechForm Industries, un acteur industriel confronté à des problèmes de contrôle qualité sur ses lignes de production. Ce cadre narratif nous a permis d’ancrer notre solution dans un scénario réaliste, d’estimer l’impact économique des erreurs de prédiction (faux positifs et faux négatifs), et de mettre en avant le retour sur investissement potentiel d’un système de détection automatisé.

    Ce projet a donc permis de mobiliser des compétences à la fois techniques (data engineering, deep learning, visualisation) et métiers (compréhension des enjeux qualité, calcul de coûts, communication des résultats), dans une optique de mise en situation professionnelle complète.
    ### Ressources à consulter :
    - Données :  
        [The MVTec anomaly detection dataset (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
        [RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection](https://github.com/hustCYQ/RAD-dataset)
    - Bibliographie :  
        [The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization](https://paperswithcode.com/paper/the-mvtec-3d-ad-dataset-for-unsupervised-3d)  
        [AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance](https://paperswithcode.com/paper/aupimo-redefining-visual-anomaly-detection)
"""
)