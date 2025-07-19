import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import pandas as pd
import cv2
from pathlib import Path
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

tab1, tab2 = st.tabs(["📋 Présentation du modèle", "🔬 Modélisation"])

with tab1:
    st.markdown("""
    # 🧠 Modèle PatchCore + WideResNet50

    Ce système permet de **détecter automatiquement des anomalies visuelles** sur des produits industriels (comme des câbles, vis, bouteilles, etc.) à partir d’images.

    ---

    ## 📦 Objectif

    Identifier :
    - ✅ les pièces **normales** (sans défaut)
    - ❌ les pièces **défectueuses** (tâche, rayure, trou, etc.)

    👉 **Sans besoin d’étiqueter les défauts** : on apprend uniquement sur des pièces "bonnes".

    ---

    ## 🧠 Comment ça fonctionne ?

    1. ### Apprentissage (sur pièces normales)
    - Le système **analyse des images de produits bons**
    - Il extrait des **"empreintes visuelles"** grâce à un réseau de neurones (**WideResNet50**)
    - Ces empreintes sont enregistrées dans une **base mémoire** (*memory bank*)

    2. ### Inspection (nouvelles images)
    - Chaque nouvelle pièce est transformée en empreinte
    - Elle est **comparée à la base mémoire**
    - Si elle est **trop différente**, elle est considérée comme **anormale**

    3. ### Seuil de décision automatique
    - Le système calcule un **score d’anomalie**
    - Il **choisit un seuil optimal** pour décider si la pièce est OK ou NOK

    4. ### Évaluation des résultats
    - 📈 Courbe ROC (pour évaluer la qualité du modèle)
    - 🔍 Matrice de confusion (visualisation des erreurs)

    ---

    ## 🔧 Technologies utilisées

    - **WideResNet50** : réseau de neurones qui extrait une représentation riche des images
    - **PatchCore** : méthode de détection d’anomalies visuelles par comparaison avec des exemples normaux
    - **Distance euclidienne** : mesure la différence entre une image et les images normales

    ---

    """)

    st.header("🧠 Comprendre le fonctionnement de WideResNet50")

    st.image(
        "static/ResNet50.png",
        caption="Architecture résumée de ResNet/WideResNet50 – Source : https://commons.wikimedia.org/wiki/File:ResNet50.png",
        use_container_width=True
    )

    st.markdown("""
    ### 🔍 Extraction de caractéristiques visuelles avec WideResNet50

    Lorsque notre système analyse une image, il ne travaille pas directement avec les pixels. Il commence par **traduire visuellement l’image en une représentation numérique plus intelligente**, appelée *vecteur de caractéristiques*.

    Pour cela, nous utilisons un réseau de neurones **pré-entraîné** très puissant : **WideResNet50**.

    ---

    ### 🧠 Pourquoi WideResNet50 ?

    WideResNet50 a été entraîné sur des millions d’images pour apprendre à détecter automatiquement :
    - des **formes**, 
    - des **textures**,
    - des **motifs visuels complexes**.

    On l’utilise ici **comme un expert qualité numérique** :
    - Il ne prend pas de décision finale,
    - Mais il **analyse l’image** et nous donne un **résumé visuel très précis** sous forme de vecteur.

    ---

    ### ⚙️ Comment ça fonctionne ?

    1. L’image passe à travers les couches internes du réseau.
    2. Certaines couches sont surveillées pour **extraire des informations intermédiaires pertinentes** (appelées *feature maps*).
    3. Ces informations sont traitées (moyennées, redimensionnées) pour former une **empreinte numérique unique** de l’image.
    4. Cette empreinte permet ensuite de :
    - **détecter les anomalies**,
    - ou comparer des images entre elles (**détection de déviation** par rapport à la norme).

    ---

    ### 🧩 En résumé

    C’est comme si on transformait chaque image en **carte d’identité visuelle**, que l’on peut ensuite comparer :
    - Une pièce "bonne" a une signature typique,
    - Une pièce "défectueuse" produit une signature différente,
    - Et c’est cette différence que l’on capte pour signaler une anomalie.

    """)

    st.markdown("""
    ### 🔎 Détail technique : extraction de caractéristiques via WideResNet50

    WideResNet50 est un réseau de neurones profond de type *ResNet* (Residual Network) mais **élargi** (*wide*), ce qui signifie que chaque couche possède plus de filtres pour mieux capturer la richesse des informations visuelles.

    Voici ce qui se passe quand une image est analysée :

    1. **Propagation dans le réseau**  
    L'image d'entrée est transformée en une série de représentations visuelles à travers plusieurs couches convolutives successives.  
    Chaque couche détecte des motifs de complexité croissante :  
    - Couches basses : contours, bords, textures simples  
    - Couches intermédiaires : formes, motifs locaux complexes  
    - Couches hautes : objets, parties d'objets, contextes visuels  

    2. **Extraction des *feature maps***  
    Au lieu d’utiliser seulement la sortie finale du réseau, on intercepte les sorties de couches intermédiaires (ici, `layer2` et `layer3`) grâce à des *hooks*.  
    Ces *feature maps* contiennent des cartes spatiales riches en informations visuelles, capturant les détails essentiels pour distinguer une pièce normale d’une pièce anormale.

    3. **Mise en forme des caractéristiques**  
    Ces *feature maps* sont moyennées spatialement (via un average pooling), puis redimensionnées pour avoir la même taille.  
    On concatène ensuite toutes ces cartes pour obtenir une **empreinte visuelle unique**.  
    Cette empreinte est un vecteur multidimensionnel représentant la "signature" visuelle de l'image, condensée et normalisée.

    4. **Utilisation dans PatchCore**  
    Cette signature est comparée à une base mémoire construite sur des pièces normales.  
    Plus la distance (différence) entre la signature de la pièce testée et celles de la mémoire est grande, plus le score d’anomalie est élevé, signalant un défaut potentiel.

    ---

    💡 **En résumé :** WideResNet50 agit comme un détective visuel expert qui transforme chaque image en une empreinte digitale visuelle complexe.  
    Cette empreinte est la clé pour détecter automatiquement les anomalies sans que le modèle ait jamais vu d’exemple de défaut.

    """)

    st.header("🧠 Comprendre le fonctionnement de PatchCore")

    st.image("static/architecture_PatchCore.png", 
             caption="Pipeline PatchCore – Source : GitHub Amazon Science (https://github.com/amazon-science/patchcore-inspection/blob/main/images/architecture.png)", 
             use_container_width =True)

    st.markdown("""
    PatchCore est une méthode de **détection d’anomalies visuelles** basée sur l’intelligence artificielle.

    Voici une explication simple de son fonctionnement :
    - Nous entraînons un modèle uniquement avec des **images normales** (produits sans défaut).
    - Ce modèle extrait les **caractéristiques visuelles importantes** de chaque image.
    - Ces caractéristiques sont stockées dans une "mémoire".
    - Lorsqu’on reçoit une **nouvelle image à inspecter**, on compare ses caractéristiques à la mémoire.
    - Si l’image test est **trop différente**, elle est considérée comme **anormale (défectueuse)**.

    Cela permet de détecter automatiquement des défauts visuels sans avoir besoin d'exemples de chaque type de défaut !
    """)

with tab2:

    st.markdown("""
<div style='border-left: 6px solid #f39c12; padding: 0.5em; background-color: #fff3cd;'>
    <strong>⚠️ Version allégée de l'application :</strong><br>
    Cette démo utilise uniquement la catégorie <code>cable</code> du dataset <strong>MVTec AD</strong>, en raison de la limite de stockage de <strong>Streamlit Community Cloud</strong> (1 Go par app).<br>
    Pour exploiter l'ensemble du dataset avec toutes les catégories, téléchargez le code complet et exécutez-le localement.
</div>
""", unsafe_allow_html=True)
    
    CATEGORIES = [
        'cable'      
    ]

    category = st.selectbox("Choisir une catégorie :", CATEGORIES, index=0)
    st.title(f"🔍 Test Anomalies MVTec - catégorie {category}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data_path = Path("../../models")
    data_path = Path(__file__).parent.parent / "models"

    if not data_path.exists():
        st.error(f"Le chemin {data_path} n'existe pas.")
        st.stop()

    # -- Chargement des fichiers --
    try:
        y_true = np.load(data_path / f"{category}_wresnet50_y_true_test.npy")
        y_score = np.load(data_path / f"{category}_wresnet50_y_scores_test.npy")
        with open(data_path / f"{category}_wresnet50_best_threshold.txt", "r") as f:
            best_threshold = float(f.read().strip())
        memory_bank = torch.load(data_path / f"{category}_wresnet50_memory_bank.pt", map_location=device)
    except Exception as e:
        st.error(f"Erreur chargement fichiers: {e}")
        st.stop()

    # -- Affichage métriques --

    # Binarisation des prédictions
    y_pred = (y_score >= best_threshold).astype(int)

    # -- ROC + Rapport de classification côte à côte
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Rapport brut
    report = classification_report(
        y_true, y_pred,
        target_names=["Good", "Anomaly"],
        output_dict=True
    )

    # Création DataFrame
    df_report = pd.DataFrame(report).transpose()

    # Convertir support en int là où applicable (sauf "accuracy")
    if "support" in df_report.columns:
        df_report["support"] = df_report["support"].apply(
            lambda x: int(x) if isinstance(x, (int, float)) and not pd.isna(x) else ""
        )

    # Mise en forme
    styled_report = df_report.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:d}"
    }).format_index(lambda idx: idx.capitalize())  # Optionnel : met des majuscules à Good/Anomaly

    # Affichage dans Streamlit

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Courbe ROC")
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.legend()
        st.pyplot(fig_roc)

    with col2:
        st.markdown("### 🔍 Matrice de confusion")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Good", "Anomaly"], yticklabels=["Good", "Anomaly"], ax=ax)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
        st.pyplot(fig_cm)


    st.markdown("### 📋 Rapport de classification")
    st.dataframe(styled_report, use_container_width=True)

    # -- Chargement du modèle backbone (WideResNet50)
    class WideResNet50FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            weights = Wide_ResNet50_2_Weights.DEFAULT
            self.model = wide_resnet50_2(weights=weights)
            self.model.eval()

            # Geler les poids
            for param in self.model.parameters():
                param.requires_grad = False

            # Enregistrer les couches pour hooks
            self.model.layer2[-1].register_forward_hook(self._hook)
            self.model.layer3[-1].register_forward_hook(self._hook)

        def _hook(self, module, input, output):
            self.features.append(output)

        def forward(self, x):
            self.features = []
            with torch.no_grad():
                _ = self.model(x)

            avg = nn.AvgPool2d(3, stride=1)
            fmap_size = self.features[0].shape[-2:]
            resize = nn.AdaptiveAvgPool2d(fmap_size)

            resized_maps = [resize(avg(fmap)) for fmap in self.features]
            patch = torch.cat(resized_maps, 1)
            return patch.reshape(patch.shape[1], -1).T  # (Nb de patches, Nb features)

    @st.cache_resource
    def load_model():
        model = WideResNet50FeatureExtractor().to(device)
        model.eval()
        return model
    

    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

        # -- Choix de la source de l'image --
    st.markdown("## 🖼️ Choisir une image de test")

    source_option = st.radio("Source de l'image :", ["📁 Image prédéfinie", "⬆️ Uploader une image"])

    image = None
    image_name = None

    if source_option == "📁 Image prédéfinie":
        
        static_image_dir = Path("static/cable")
        image_paths = list(static_image_dir.glob("*/*.jpg")) + list(static_image_dir.glob("*/*.png"))
  
        if not image_paths:
            st.warning("Aucune image trouvée dans `static/cable/*/*.jpg|png`.")
            st.stop()

        # Crée une liste d'options affichant la classe et le nom de l'image
        image_options = [f"{p.parent.name} / {p.name}" for p in image_paths]
        selected_image_display = st.selectbox("Sélectionner une image :", image_options)

        # Récupère le chemin réel à partir du nom choisi
        selected_image_index = image_options.index(selected_image_display)
        selected_image_path = image_paths[selected_image_index]
        selected_class = selected_image_path.parent.name
        image = Image.open(selected_image_path).convert("RGB")
        image_name = selected_image_path.name

        st.markdown(f"**Classe de l'image sélectionnée : `{selected_class}`**")


    elif source_option == "⬆️ Uploader une image":
        uploaded_file = st.file_uploader("Uploader une image (jpg/png/jpeg)")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    if image is not None:

        # -- Prédiction --
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(input_tensor)
            distances = torch.cdist(features, memory_bank, p=2)
            dist_score, _ = torch.min(distances, dim=1)
            s_star = torch.max(dist_score)
            segm_map = dist_score.view(1, 1, 28, 28)
            segm_map = torch.nn.functional.interpolate(segm_map, size=(224, 224), mode="bilinear")
            heat_map = segm_map.squeeze().cpu().numpy()
            anomaly_score = s_star.item()
            pred = int(anomaly_score >= best_threshold)
            label = ['✅ OK', '❌ Anomaly'][pred]

        st.markdown("### Résultat prédiction")
        st.write(f"Score d'anomalie : **{anomaly_score:.2f}**")
        st.write(f"Seuil utilisé : **{best_threshold:.2f}**")
        st.write(f"Prédiction : **{label}**")

        # Création de la superposition heatmap + image originale
        original_np = np.array(image.resize((224, 224))).astype(np.uint8)

        heatmap_norm = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        superposed = cv2.addWeighted(cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6,
                                    heatmap_color, 0.4, 0)
        superposed_rgb = cv2.cvtColor(superposed, cv2.COLOR_BGR2RGB)

        # Affichage en grille 2×2
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Image originale")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(superposed_rgb)
        axs[0, 1].set_title("Superposition heatmap + image")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(heat_map, cmap='jet')
        axs[1, 0].set_title(f"Carte de chaleur anomalie\n(score: {anomaly_score:.2f})")
        axs[1, 0].axis("off")

        seuils = [best_threshold, best_threshold * 1.2, best_threshold * 1.5]
        segm_multi = np.zeros_like(heat_map, dtype=np.uint8)
        for i, seuil in enumerate(seuils, start=1):
            segm_multi += (heat_map > seuil).astype(np.uint8)

        axs[1, 1].imshow(segm_multi, cmap='viridis', vmin=0, vmax=len(seuils))
        axs[1, 1].set_title("Segmentation multi-niveaux")
        axs[1, 1].axis("off")

        st.pyplot(fig)