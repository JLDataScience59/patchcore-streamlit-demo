import streamlit as st

from utils.style import apply_custom_css
from utils.sidebar import render_sidebar

def main():
    render_sidebar()

if __name__ == '__main__':
    main()

apply_custom_css()

st.title("🙏 Remerciements")

st.markdown("""
Ce projet a été une étape marquante dans notre parcours de formation, mêlant rigueur technique, travail d’équipe et application concrète des compétences acquises. Nous souhaitons adresser nos sincères remerciements aux personnes et structures qui nous ont accompagnés.

- 👨‍🏫 À nos **intervenants pédagogiques**, pour la qualité de leur encadrement, leur exigence constructive et leur disponibilité tout au long du projet. Leur expertise a grandement enrichi notre réflexion.

- 🏫 À **DataScientest**, pour avoir conçu un programme complet, orienté vers la pratique, et pour les conditions d’apprentissage favorables qu’il nous a offertes.

- 👥 À nous mêmes, collègues de projet, **Jérémy Lesot, Éric Mathieu, Alexandre Mathieu et Roberto Vercellin**, pour notre implication, notre professionnalisme et l'excellente collaboration dont nous avons bénéficié tout au long de cette aventure.

Ce projet est le fruit d’un engagement collectif et d’un environnement propice à l’apprentissage. Qu’il s’agisse de partager des idées, de surmonter des obstacles techniques ou de construire une solution robuste, chaque acteur a contribué à sa réussite.
""", unsafe_allow_html=True)