# Détection d’Anomalies Visuelles avec PatchCore + WideResNet50

Cette application Streamlit permet de détecter automatiquement des anomalies visuelles sur des images de produits industriels (ex : câbles, vis, bouteilles) en utilisant un modèle PatchCore basé sur un backbone WideResNet50.

---

## Lien vers l'application

<a href="https://patchcore-app-demo-6chnpfpspezhmarj9o8hjt.streamlit.app/">defectAI</a>

---

## Fonctionnalités

- Présentation du modèle et de la méthode PatchCore
- Visualisation des performances (courbe ROC, matrice de confusion, rapport de classification)
- Test interactif : upload d’image pour prédiction d’anomalie et affichage d’une heatmap d’anomalie

---

## Structure du projet

- `app.py` : code principal de l’application Streamlit  
- `utils/` : fonctions utilitaires (style CSS, sidebar)  
- `models/` : fichiers de modèles et données nécessaires (memory bank, scores, etc.)  
- `static/` : images statiques pour l’interface  
- `pages/` : pages de l’application  
- `style/` : fichiers de styles CSS pour l'application

---

## Installation locale

1. Cloner ce dépôt

```bash
git clone https://github.com/JLDataScience59/patchcore-streamlit-demo.git
cd patchcore-streamlit-demo
```

2. Créer un environnement virtuel Python (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. Lancer l’application

```bash
streamlit run app.py
```

## Notes
- Cette version démo utilise uniquement la catégorie cable du dataset MVTec AD, pour respecter la limite de stockage de Streamlit Community Cloud.
- Pour une version complète, téléchargez le code et les datasets complets et exécutez l’app localement.

## Licence
MIT License
