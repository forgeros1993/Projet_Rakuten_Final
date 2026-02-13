"""
Page Explicabilité des Modèles.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from PIL import Image

# --- GESTION DES CHEMINS ROBUSTE ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config import APP_CONFIG, ASSETS_DIR
    from utils.ui_utils import load_css
except ImportError:
    APP_CONFIG = {"title": "Rakuten", "layout": "wide"}
    ASSETS_DIR = Path(".")

st.set_page_config(
    page_title=f"Explicabilité - {APP_CONFIG['title']}",
    page_icon="🔬",
    layout=APP_CONFIG["layout"],
)

if (ASSETS_DIR / "style.css").exists():
    load_css(ASSETS_DIR / "style.css")

# Header
st.title("Explicabilité & Transparence")
st.markdown("Ouvrir la 'Boîte Noire' : Comprendre pourquoi l'IA prend une décision.")

# Méthodes
st.divider()
st.header("Méthodes par Modalité")

col1, col2, col3, col4 = st.columns(4)
col1.metric("SHAP", "Texte", "Impact mot à mot")
col2.metric("Attention Maps", "Image (ViT)", "Focus Sémantique")
col3.metric("Grad-CAM", "Image (CNN)", "Focus Texture")
col4.metric("Features", "XGBoost", "Statistiques")

# --- SECTION TEXTE (SHAP) ---
st.divider()
st.header("1. Explicabilité Texte (SHAP)")
st.info("Produit analysé : **iPhone 15 Pro Max 256GB Smartphone Apple**")

shap_data = pd.DataFrame({
    'Token': ['iphone', 'smartphone', 'apple', '256gb', 'pro', 'max'],
    'Impact (+/-)': [0.45, 0.35, 0.20, 0.10, 0.05, 0.02],
})

fig_shap = px.bar(shap_data, x='Impact (+/-)', y='Token', orientation='h',
                  color='Impact (+/-)', color_continuous_scale=['#FFE5E5', '#BF0000'],
                  title="Contribution positive vers la classe 'Téléphonie'")
fig_shap.update_layout(height=300, coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)

# --- SECTION IMAGE ---
st.divider()
st.header("2. Explicabilité Image (Architecture Voting)")
st.markdown("Le modèle Image est un **trio** pondéré (4-2-1).")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1. DINOv3")
    st.info("🧠 **Poids : 4**")
    st.metric("Score", "79.4%")
    st.caption("Vision Globale & Formes")
with c2:
    st.subheader("2. XGBoost")
    st.info("⚖️ **Poids : 2**")
    st.metric("Score", "85.3%*")
    st.caption("*Sur features internes")
with c3:
    st.subheader("3. EfficientNet")
    st.info("👀 **Poids : 1**")
    st.metric("Score", "66.6%")
    st.caption("Textures & Détails")

st.success("💡 **Synergie** : DINOv3 donne la direction générale, XGBoost valide statistiquement, EfficientNet gère les détails de texture.")

# --- section image heatmap ---
st.write("---")
st.subheader("Visualisation des Zones d'Interet")

# je definis les chemins possibles
# le dossier actuel (pages) et le dossier assets
dir_actuel = Path(__file__).parent
dir_assets = dir_actuel.parent / "assets"

chemins_test = [
    dir_actuel / "heatmap_demo.png",
    dir_assets / "heatmap_demo.png"
]

trouve = False
for p in chemins_test:
    if p.exists():
        try:
            # je charge et j affiche
            image_cam = Image.open(p)
            st.image(image_cam, caption="Comparaison : Attention (DINO) vs Activation (EffNet)", use_container_width=True)
            trouve = True
            break
        except Exception as e:
            st.error(f"erreur lecture fichier : {e}")

if not trouve:
    st.warning("Fichier heatmap_demo.png non detecte")
    # je t affiche les chemins testes pour debug
    st.write("Chemins verifies :")
    for p in chemins_test:
        st.code(str(p))

# --- SECTION FUSION (MISE A JOUR STRATÉGIQUE) ---
st.divider()
st.header("3. Explicabilité de la Fusion")

# NOUVEAUX POIDS EQUILIBRÉS
w_img_opt = 0.5
w_txt_opt = 0.5

c_eq, c_expl = st.columns([2, 1])

with c_eq:
    st.subheader("Formule de Décision")
    st.latex(r"Score_{Final} = (Score_{Image} \times 0.5) + (Score_{Texte} \times 0.5)")
    st.markdown("**Configuration Équilibrée** :")
    st.progress(w_img_opt, text=f"Poids Image : {w_img_opt:.0%}")
    st.progress(w_txt_opt, text=f"Poids Texte : {w_txt_opt:.0%}")

with c_expl:
    st.info("""
    **Pourquoi 50/50 ?**
    - **Texte (84%)** : Très précis mais échoue si la description est vide/courte.
    - **Image (79%)** : Robuste, capte ce que le texte ne dit pas (couleurs, formes).
    
    L'équilibre parfait garantit la meilleure robustesse.
    """)
    st.caption("⚠️ *Ce paramètre est ajustable en temps réel dans l'onglet Démo.*")

# Sidebar
with st.sidebar:
    st.markdown("### Transparence")
    st.divider()
    st.info("Architecture validée")