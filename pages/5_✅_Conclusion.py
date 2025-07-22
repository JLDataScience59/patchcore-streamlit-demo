import streamlit as st

from utils.style import apply_custom_css
from utils.sidebar import render_sidebar

def main():
    render_sidebar()

if __name__ == '__main__':
    main()

apply_custom_css()

st.title("✅ Conclusion")

st.markdown("""
Au terme de ce projet, nous avons pu démontrer qu’un système de détection d’anomalies industrielles automatisé, fiable et explicable est non seulement envisageable, mais aussi performant lorsque des approches avancées de deep learning sont mobilisées.
Notre démarche itérative — partant de modèles classiques de machine learning, jusqu’à l’implémentation de solutions SOTA comme **PatchCore avec WideResNet50** — nous a permis de :  
            
- Valider l’intérêt d’une phase exploratoire avec des modèles simples, utiles pour comprendre la structure des données.
- Constater les limites critiques de ces modèles dans des contextes industriels complexes, marqués par l’hétérogénéité des défauts et une exigence élevée de rappel.
- Évaluer et affiner des architectures de deep learning, avec un accent particulier sur les autoencodeurs pour la détection non supervisée, puis sur les CNN pour la classification supervisée.
- Identifier **PatchCore** comme la méthode la plus robuste et la plus opérationnelle pour une industrialisation à grande échelle, notamment grâce à son efficacité sans besoin de données défectueuses et sa capacité à se généraliser à des défauts inconnus.

Ce projet nous a également permis de créer une **application Streamlit** fonctionnelle, illustrant de manière concrète l’intégration d’un système de détection dans une interface simple, compréhensible et exploitable pour un utilisateur métier. Cette application permet de charger une image de pièce industrielle et d’en obtenir une analyse visuelle accompagnée d’une décision (anomalie détectée ou non), tout en mettant à disposition des indicateurs explicables.
En somme, nous avons posé les bases d’une solution industrialisable, combinant performance technique, interprétabilité et ergonomie d’usage, capable de répondre aux enjeux stratégiques d’une entreprise comme Techform Industries.
""", unsafe_allow_html=True)

st.markdown("""
---
<small><em>Note : TechForm Industries est une société fictive utilisée à des fins pédagogiques.</em></small>
""", unsafe_allow_html=True)