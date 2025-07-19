import streamlit as st
from utils.style import apply_custom_css
from utils.sidebar import render_sidebar

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

st.title("🏭 Simulation métier : TechForm Industries")
st.info("ℹ️ **TechForm Industries** est une entreprise fictive créée pour ce projet pédagogique. Son nom, son logo et son activité ont été générés à l’aide d’outils d’intelligence artificielle.")

col1, col2, col3 = st.columns([1, 2, 1])  # Ratio : 25% / 50% / 25%

with col2:
    st.image("static/techform.png", use_container_width=True, caption="Logo généré pour l’entreprise fictive TechForm Industries")

tab1, tab2 = st.tabs(["📋 Présentation", "📌 Conclusion"])

with tab1:
    st.markdown(
        """
        ### 1.Contexte  
        **TechForm Industries** est une entreprise spécialisée dans la fabrication de composants plastiques moulés et injectés destinés à des secteurs sensibles comme l’automobile, le médical ou l’électronique grand public. Dans un contexte d’augmentation des volumes et de diversification des gammes de produits, l’entreprise souhaite **renforcer son contrôle qualité automatisé** en fin de chaîne de production.
        Récemment, plusieurs incidents ont mis en évidence des défauts visuels non détectés (fissures, bavures, déformations), occasionnant des **retours clients coûteux**, des arrêts de production chez des partenaires, voire des risques réglementaires dans les secteurs critiques.
        Ne disposant pas encore d’un historique d’images annotées propre à ses pièces, TechForm nous a missionnés pour **prototyper une solution de détection automatique de défauts visuels** à partir d’un dataset proche de leur environnement réel.  
        ### 2.Jeu de données de substitution  
        Pour ce cas d’usage, nous utilisons le **dataset public MVTEC AD** qui propose des images haute résolution de pièces industrielles comportant ou non des défauts visuels. Plusieurs objets de ce dataset correspondent à des formes et textures proches des produits de TechForm :  
        
        - bottle, capsule : pièces plastiques moulées  
        - screw : petits composants métalliques  
        - tile, wood, grid : surfaces structurées utilisées dans d’autres segments  
        
        Le dataset fournit aussi, pour chaque image défectueuse, un **masque segmentant précisément la zone défectueuse**, ouvrant la voie à des modèles de classification ou de segmentation.
        ### 3.Problématique métier et modélisation  
        L’objectif principal est de construire un modèle capable de :  
        1.	**Détecter automatiquement si une pièce est défectueuse** (classification binaire)  
        2.	**Identifier le type de défaut** (multi-classes)  
        3.	(Éventuellement) **localiser la zone défectueuse** (segmentation via masques)  

        Cependant, le projet s’inscrit dans un contexte métier particulier avec des contraintes économiques fortes liées aux erreurs du modèle.  
        ### 4.Coûts associés aux erreurs  
        | Type d’erreur      | Description                                 | Coût estimé | Volume estimé en 2024 | Coût total en 2024 |
        |--------------------|---------------------------------------------|-------------|------------------------|---------------------|
        | Faux négatif (FN)  | Défaut non détecté ➜ pièce défectueuse livrée | **15 €**        | **~ 2 500 pièces**         | **37 500 €**            |
        | Faux positif (FP)  | Bonne pièce rejetée à tort                  | **3 €**         | **~ 4 000 pièces**         | **12 000 €**            |

        💶	**Coût total annuel 2024 des erreurs : 49 500 €**  

        Ce chiffre représente une perte directe (retours, recontrôles, logistique) que l’entreprise souhaite fortement réduire.  
        L’objectif fixé est de diviser ce coût **par 4**, en abaissant le taux de faux négatifs notamment, soit une **économie annuelle cible de plus de 35 000 €**.  
        Le client insiste sur le fait qu’il est **préférable de rejeter une pièce saine** plutôt que de laisser passer un défaut non détecté. L’objectif est donc d’avoir un **appel** élevé, quitte à réduire légèrement la précision.  

        ### 5.Conséquences sur le modèle  
        
        Notre solution intègre donc les éléments suivants :  

        - Choix de métriques de performance métier-compatibles : **Recall, F1-score, matrice de confusion pondérée**  
        - Ajustement du **seuil de classification** pour optimiser le rappel  
        - Possibilité d’introduire une **fonction de perte asymétrique** ou un **coût personnalisé** dans la phase d’entraînement  

        ### 6.Perspectives  

        Une fois ce prototype validé, TechForm envisage d’intégrer la solution à sa ligne de production, avec :  

        - un module de prise de vue synchronisé à la cadence de fabrication,  
        - un modèle embarqué sur GPU local,  
        - et un système de tri automatique basé sur la sortie du modèle.  

    """
    )

with tab2:
    st.markdown(
    """
    ### 1.Estimation dynamique du coût des erreurs   
    🔹 Paramètres d’entrée  

    - N : nombre total de pièces traitées en 2024 (ex. 80 000)  
    - P_defect : proportion réelle d’objets défectueux (ex. 8%)  
    - R = recall du modèle  
    - P = precision du modèle  

    🔹 Calculs intermédiaires  

    - Nombre de pièces défectueuses : Ndefect = N × Pdefect  
    - Faux négatifs (FN) : FN = Ndefect × (1−recall)  
    - Faux positifs (FP) : FP = (TP/precision) − TP où TP = Ndefect × recallFP   

    🔹 Coût métier total  
    
    - C_FN = 15 € (coût par faux négatif)  
    - C_FP = 3 € (coût par faux positif)  

        Coût total = FN × CFN + FP × CFP   

    ### 2.Exemple chiffré 
    | Paramètre      | Valeur                                 |
    |--------------------|---------------------------------------------|
    | Nombre total (N)  | 80 000 pièces |
    | Proportion défectueux  | 8% |
    | Recall du modèle  | 0.94 |  
    | Précision du modèle  | 0.78 |  
    | Coût par FN  | 15€ |  
    | Coût par FP  | 3€ |   

    Calcul :
    - Ndefect = 6 400
    - FN = 6 400 × (1 − 0.94) = 384
    - TP = 6 016
    - FP = 6 016 / 0.78 − 6 016=1 697
    💰 Coût total = 384×15 + 1 697×3 = 5 760 + 5 091 = 10 851 €

    🔻 Économie annuelle estimée :
    49 500€ − 10 851€ = 38 649 € économisés
  
    ### 3.Visualisation des résultats  
    Les graphiques suivants illustrent visuellement l’impact économique du modèle proposé :  
    
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.image("static/final_report1.png", use_container_width=True, caption="Figure 1 – Comparaison des coûts annuels (avec vs sans modèle)")
   
    with col2:
        st.image("static/final_report2.png", use_container_width=True, caption="Figure 2 – Matrice de confusion pondérée")

    st.markdown("""
    ### 4.Commentaires métiers 

    Avec un modèle performant (recall > 90%), TechForm Industries pourrait **réduire de plus de 80%** le coût annuel lié aux défauts non détectés. Même avec un volume non négligeable de faux positifs, **le gain global est significatif**, et justifie l’intégration d’un système de détection automatisé.            
    """
    )

st.markdown("""
---
<small><em>Note : TechForm Industries est une société fictive utilisée à des fins pédagogiques. Le logo présenté a été généré par une IA et ne représente aucune entité réelle.</em></small>
""", unsafe_allow_html=True)
