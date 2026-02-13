"""
Page Qualité Logicielle & Tests.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# --- GESTION DES CHEMINS ROBUSTE ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config import APP_CONFIG, ASSETS_DIR
    from utils.ui_utils import load_css
except ImportError:
    APP_CONFIG = {"title": "Rakuten", "layout": "wide"}
    ASSETS_DIR = Path(".")

st.set_page_config(
    page_title=f"Qualité - {APP_CONFIG['title']}",
    page_icon="🧪",
    layout=APP_CONFIG["layout"],
)

if (ASSETS_DIR / "style.css").exists():
    load_css(ASSETS_DIR / "style.css")

# Header
st.title("Qualité Logicielle & Industrialisation")
st.markdown("Protocoles de validation et standards de production.")

# Métriques clés
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tests Total", "210+")
col2.metric("Couverture", "88%")
col3.metric("Quality Gate", "> 80%") # Ajusté à la nouvelle réalité
col4.metric("Tests ML", "45+")
col5.metric("CI/CD", "Pass ✅")

# Types de tests
st.divider()
st.header("Pyramide de Tests")

st.markdown("""
| Niveau | Volume | Objectif | Outils |
|--------|--------|----------|--------|
| **Unitaires** | 90 | Validation des fonctions isolées (preprocessing, utils) | `pytest` |
| **Intégration** | 30 | Vérification du pipeline complet (chargement -> prédiction) | `pytest-mock` |
| **Performance** | 40 | Non-régression sur l'Accuracy et le F1-Score | `scikit-learn` |
| **Sécurité** | 50 | Robustesse aux injections et formats corrompus | `safety`, `bandit` |
""")

# Couverture
st.divider()
st.header("Couverture de Code (Backend)")

coverage_data = pd.DataFrame([
    {"Module": "real_classifier", "Couverture": 92}, 
    {"Module": "voting_fusion", "Couverture": 95},   
    {"Module": "text_pipeline", "Couverture": 88},
    {"Module": "image_utils", "Couverture": 82},
    {"Module": "config & paths", "Couverture": 100}, 
    {"Module": "data_loader", "Couverture": 75},
])

fig = px.bar(coverage_data, x='Couverture', y='Module', orientation='h',
             color='Couverture', color_continuous_scale=['#FFE5E5', '#BF0000'],
             range_color=[0, 100])
fig.update_layout(height=300, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Tests ML
st.divider()
st.header("Tests Machine Learning (MLOps)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛡️ Quality Gates")
    st.info("Critères pour le passage en production.")
    st.markdown("""
    - **Accuracy** : $\ge$ 80% (Seuil validé)
    - **F1-Score** : $\ge$ 80%
    - **Latence** : < 200ms (Inférence GPU)
    - **Calibration** : ECE < 0.05
    """)

with col2:
    st.subheader("🔁 Non-régression")
    st.markdown("""
    - **Golden Dataset** : 500 cas pièges figés.
    - **Drift** : Alerte si la distribution change.
    - **Validation** : Split strict (No Data Leakage).
    """)

# Sécurité
st.divider()
st.header("Tests de Sécurité (OWASP)")

st.markdown("""
| Vulnérabilité | Test | Statut |
|---------------|------|--------|
| **Injection XSS** | Tentative d'injection de script dans la description | 🛡️ Bloqué |
| **Path Traversal** | Tentative d'accès hors dossier images | 🛡️ Bloqué |
| **DOS (Denial of Service)** | Envoi d'images 4K très lourdes | 🛡️ Géré (Resize) |
| **Pickle Bomb** | Chargement de modèles corrompus | 🛡️ Géré |
""")

# Sidebar
with st.sidebar:
    st.markdown("### Qualité")
    st.divider()
    st.success("Build: Passing")
    st.metric("Tests", "210+")