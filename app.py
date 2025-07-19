# app.py
import streamlit as st
from utils.style import apply_custom_css
from utils.sidebar import render_sidebar
from pathlib import Path
import os

apply_custom_css()
render_sidebar()

st.title("Détection d'anomalies industrielles par PatchCore")
#st.markdown("Bienvenue dans notre projet Streamlit ! Naviguez via le menu à gauche.")

st.markdown("""
<div style='border-left: 6px solid #f39c12; padding: 0.5em; background-color: #fff3cd;'>
    <strong>⚠️ Version allégée de l'application :</strong><br>
    Cette démo utilise uniquement la catégorie <code>cable</code> du dataset <strong>MVTec AD</strong>, en raison de la limite de stockage de <strong>Streamlit Community Cloud</strong> (1 Go par app).<br>
    Pour exploiter l'ensemble du dataset avec toutes les catégories, téléchargez le code complet et exécutez-le localement.<br>       
    Cette version de l'application est une première ébauche, tant au niveau du contenu scientifique (partie modélisation) que du contenu fonctionnel (partie simulation et ergonomire de l'application).<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='border-left: 6px solid #f39c12; padding: 0.5em; background-color: #fff3cd;'>
    <strong>Cette application est une première ébauche réalisée dans le cadre de ma formation en data science.<br>
    Elle a pour objectif de mettre en avant le travail effectué sur la détection d’anomalies industrielles,<br>
    en combinant modélisation, traitement d’images et déploiement d’une interface interactive.</strong><br><br>
</div>
""", unsafe_allow_html=True)
