"""
Page de présentation du pipeline de preprocessing.
"""
import streamlit as st
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Preprocessing - {APP_CONFIG['title']}",
    page_icon="⚙️",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Pipeline de Preprocessing")

# Métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Produits", "84 916")
col2.metric("Vocabulaire TF-IDF", "~15K")
col3.metric("Features Image", "2 048")
col4.metric("Langues", "5")

# Pipeline Texte
st.divider()
st.header("Pipeline Texte")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Nettoyage")
    st.markdown("""
    - Suppression HTML
    - Caractères spéciaux
    - Normalisation espaces
    """)

with col2:
    st.subheader("2. Langue")
    st.markdown("""
    - Détection (langid)
    - Traduction → FR
    - ~85% déjà en français
    """)

with col3:
    st.subheader("3. Vectorisation")
    st.markdown("""
    - TF-IDF
    - ~15K dimensions
    - designation + description
    """)

# Pipeline Image
st.divider()
st.header("Pipeline Image")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Resize")
    st.markdown("""
    - 224 × 224 pixels
    - Format attendu ResNet
    """)

with col2:
    st.subheader("2. Normalisation")
    st.markdown("""
    - Mean/std ImageNet
    - [0.485, 0.456, 0.406]
    """)

with col3:
    st.subheader("3. Extraction")
    st.markdown("""
    - ResNet50 pré-entraîné
    - 2048 features (avg_pool)
    """)

# Choix techniques
st.divider()
st.header("Choix Techniques")

tab_text, tab_image = st.tabs(["Texte", "Image"])

with tab_text:
    st.markdown("""
    | Choix | Justification |
    |-------|---------------|
    | TF-IDF (pas Word2Vec) | Performance équivalente, meilleure interprétabilité |
    | Traduction FR | 85% FR, vocabulaire unifié |
    | Pas de lemmatisation | Préserver marques (iPhone, PlayStation) |
    """)

with tab_image:
    st.markdown("""
    | Choix | Justification |
    |-------|---------------|
    | ResNet50 (pas VGG16) | Meilleur compromis taille/performance |
    | avg_pool (pas flatten) | Moins de dimensions, moins d'overfitting |
    | Pas d'augmentation | Dataset assez grand (85K) |
    """)

# Démo
st.divider()
st.header("Démo")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    demo_text = st.text_area(
        "Texte brut",
        value="<p>iPhone 15 Pro Max</p> - Smartphone Apple, écran OLED 6.7\"",
        height=100
    )

with col2:
    st.subheader("Output")
    if demo_text:
        cleaned = re.sub(r'<[^>]+>', '', demo_text)
        cleaned = re.sub(r'[^\w\s\-]', ' ', cleaned)
        cleaned = ' '.join(cleaned.lower().split())
        st.code(cleaned)
        st.caption(f"{len(cleaned)} caractères, {len(cleaned.split())} mots")

# Sidebar
with st.sidebar:
    st.markdown("### Preprocessing")
    st.divider()
    st.markdown("""
    **Texte**: Nettoyage → Langue → TF-IDF

    **Image**: Resize → Norm → ResNet50
    """)
