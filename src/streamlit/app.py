"""
Application Streamlit - Classification de produits Rakuten.
"""
import streamlit as st
import sys
from pathlib import Path

# --- GESTION DES CHEMINS ROBUSTE ---
# Ajout de la racine du projet au path pour trouver 'config' et 'utils'
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APP_CONFIG, MODEL_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css
from utils.category_mapping import get_all_categories

# Configuration de la page
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"],
)

# Chargement du style CSS
if (ASSETS_DIR / "style.css").exists():
    load_css(ASSETS_DIR / "style.css")

# --- INITIALISATION DU CLASSIFIEUR (SESSION STATE) ---
if "classifier" not in st.session_state:
    with st.spinner("Chargement des cerveaux numériques (DINO + XGBoost + EfficientNet)..."):
        from utils.real_classifier import MultimodalClassifier
        try:
            # On instancie le vrai classifieur
            st.session_state.classifier = MultimodalClassifier()
            st.session_state.use_mock = False
        except Exception as e:
            st.error(f"Echec chargement modèle : {e}")
            # En cas d'erreur critique, on charge le mock pour ne pas planter l'app
            from utils.mock_classifier import DemoClassifier
            st.session_state.classifier = DemoClassifier()
            st.session_state.use_mock = True

if "use_mock" not in st.session_state:
    st.session_state.use_mock = MODEL_CONFIG["use_mock"]

# --- HEADER & TITRE ---
st.title("Rakuten Product Classifier")
st.markdown("Classification automatique de produits e-commerce en 27 catégories.")

st.divider()

# Métriques clés
col1, col2, col3, col4 = st.columns(4)
col1.metric("Produits", "84 916")
col2.metric("Catégories", "27")
col3.metric("Modalités", "Texte + Image")
col4.metric("Précision", "86.5%")

st.divider()

# Contexte
st.header("Le Projet")
st.markdown("""
Rakuten France traite des millions de produits chaque année.
Ce projet automatise la classification par Deep Learning multimodal,
combinant analyse de texte (désignation, description) et d'image.
""")

# Pipeline
st.header("Pipeline")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Texte")
    st.markdown("TF-IDF + LinearSVC")

with col2:
    st.subheader("Image")
    st.markdown("Voting : DINOv3 + XGBoost + EfficientNet")

with col3:
    st.subheader("Fusion")
    st.markdown("Pondération Dynamique (50/50)")

st.divider()

# Catégories (grille compacte)
st.header("27 Catégories")
try:
    categories = get_all_categories()
    cat_list = list(categories.items())

    for i in range(0, 27, 9):
        cols = st.columns(9)
        for j, col in enumerate(cols):
            if i + j < len(cat_list):
                code, (name, full_name, emoji) = cat_list[i + j]
                col.markdown(f"{emoji} **{name}**")
except:
    st.info("Chargement des catégories en cours...")

st.divider()

# Navigation
st.header("Tester")
col1, col2 = st.columns(2)
with col1:
    if st.button("Classifier un produit", use_container_width=True, type="primary"):
        st.switch_page("pages/4_🔍_Démo.py")
with col2:
    if st.button("Explorer les données", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📊_Données.py")

# Footer
st.divider()
st.caption("Projet DataScientest — Formation BMLE Octobre 2025")

# Sidebar
with st.sidebar:
    st.markdown("### Rakuten")
    st.markdown("Product Classifier")
    st.divider()
    if st.session_state.use_mock:
        st.warning("Mode Démo (Mock)")
    else:
        st.success("Production (GPU)")