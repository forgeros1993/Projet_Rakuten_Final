import streamlit as st
import time
import sys
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# hack pour trouver les modules du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- IMPORTS AJOUTÉS POUR LE STYLE BLANC ---
from config import ASSETS_DIR
from utils.ui_utils import load_css
# -------------------------------------------

from utils.real_classifier import MultimodalClassifier

st.set_page_config(page_title="Demonstration Rakuten", layout="wide")

# --- CHARGEMENT DU STYLE (Force le fond blanc) ---
load_css(ASSETS_DIR / "style.css")
# -------------------------------------------------

st.title("Demonstration Interactive et Explicabilite")
st.markdown("---")

# chargement unique du cerveau
@st.cache_resource
def get_clf():
    return MultimodalClassifier()

clf = get_clf()

# --- fonctions utilitaires ---

def apply_stress_test(image_path):
    # je charge l'image avec pil
    img = Image.open(image_path)
    # conversion en tableau numpy
    img_array = np.array(img)
    # generation de bruit aleatoire (gaussien)
    noise = np.random.normal(loc=0, scale=25, size=img_array.shape)
    # addition image + bruit et bornage entre 0 et 255
    noisy_img = np.clip(img_array + noise, 0, 255).astype('uint8')
    # retour en image pil
    return Image.fromarray(noisy_img)

def show_traffic_light(results):
    # on recupere le top 1 et top 2 pour analyser l'ecart
    top1 = results[0]
    score = top1['confidence']
    
    # par defaut incertain
    status = "INCERTAIN"
    color = "orange"
    msg = "Le modèle hésite. Validation humaine recommandée."
    icon = "🟠"
    
    # calcul du gap (ecart) avec le 2eme
    gap = 0
    if len(results) > 1:
        top2 = results[1]
        gap = score - top2['confidence']
    
    # --- NOUVELLE LOGIQUE SOUPLE ---
    # regle 1 : score tres haut (absolu)
    if score >= 0.70:
        status = "AUTOMATISATION"
        color = "green"
        msg = "Confiance absolue (>70%). Validation automatique."
        icon = "🟢"
        
    # regle 2 : score moyen mais gros ecart avec le 2eme (relatif)
    # ex: top1=55%, top2=10% -> c'est vert car il n'y a pas de concurrent serieux
    elif score >= 0.40 and gap >= 0.20:
        status = "AUTOMATISATION (MARGE FORTE)"
        color = "green"
        msg = f"Score moyen ({score:.0%}) mais sans concurrence (Ecart +{gap:.0%})."
        icon = "🟢"
        
    # regle 3 : score trop bas
    elif score < 0.20:
        status = "REJET"
        color = "red"
        msg = "Score trop proche du hasard (3%)."
        icon = "🔴"

    # affichage
    st.markdown("### 🚦 Verdict Opérationnel")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f"<h1 style='text-align: center;'>{icon}</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Décision :** <span style='color:{color}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
        st.caption(msg)

        
def explain_pipeline_text(text):
    # affichage des etapes internes du nlp
    st.markdown("#### Traçage du Pipeline NLP")
    
    # etape 1 raw
    st.text(f"1. Input Brut : {text[:50]}...")
    
    # etape 2 clean
    clean = text.lower().replace(",", "").replace(".", "")
    st.text(f"2. Nettoyage  : {clean[:50]}...")
    
    # etape 3 token
    tokens = clean.split()
    st.text(f"3. Tokenizing : {tokens[:5]}...")
    
    # etape 4 vector
    st.text("4. Vectorisation : TF-IDF (Poids statistiques calcules)")
    
    # etape 5 model
    st.text("5. Classification : LinearSVC (Separateur lineaire)")

def show_results(results):
    # verif si resultat existe
    if not results:
        st.error("Erreur : Pas de resultat.")
        return
    
    top = results[0]
    
    # affichage principal du gagnant
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Prediction", top['label'])
    with c2:
        st.success(f"Categorie : {top['name']}")
        st.progress(top['confidence'])
        st.caption(f"Score de Confiance : {top['confidence']:.2%}")
        
    st.markdown("---")
    st.markdown("#### Distribution des Probabilites (Top 5)")
    
    # on recupere les 5 meilleurs pour le contexte
    top_n = results[:5]
    
    # creation dataframe pour affichage propre
    data = {
        "Categorie": [x['name'] for x in top_n],
        "Confiance": [x['confidence'] for x in top_n]
    }
    df = pd.DataFrame(data)
    
    # index sur la categorie pour le graphe
    df = df.set_index("Categorie")
    
    # affichage du graphique en barres
    st.bar_chart(df)
    
    # on explique le graphique
    st.caption("Ce graphique montre l'ecart entre la classe gagnante et les autres.")
    
    st.markdown("---")
    
    # appel feu tricolore corrige
    # on envoie toute la liste results et pas juste le score
    show_traffic_light(results)

# --- interface principale ---

tabs = st.tabs(["Analyse Texte", "Analyse Image", "FUSION Multimodale"])

# ==========================================
# onglet 1 : texte
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entree Texte")
        txt_input = st.text_area("Description du produit", height=200, 
                                 placeholder="Ex: Piscine gonflable pour enfants intex...")
        btn_txt = st.button("Lancer Analyse Texte", type="primary")
    
    with col2:
        st.subheader("Resultats Sémantiques")
        if btn_txt and txt_input:
            with st.spinner("Decodage des vecteurs..."):
                res = clf.predict_text(txt_input)
                
                # explicabilite
                with st.expander("Voir le raisonnement interne (Pipeline)", expanded=True):
                    explain_pipeline_text(txt_input)
                
                show_results(res)

# ==========================================
# onglet 2 : image
# ==========================================
with tabs[1]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Entree Image")
        img_file = st.file_uploader("Fichier image", type=['jpg', 'png', 'jpeg'])
        
        # option stress test
        use_stress = st.checkbox("Activer Stress Test (Bruit Numerique)")
        
        if img_file:
            # sauvegarde initiale
            with open("temp_demo.jpg", "wb") as f: 
                f.write(img_file.getbuffer())
            
            # affichage conditionnel
            if use_stress:
                # generation image bruitee
                noisy = apply_stress_test("temp_demo.jpg")
                st.image(noisy, caption="Image Modifiee (Simulation Mauvaise Qualite)", use_container_width=True)
                # ecrasement temporaire pour la prediction
                noisy.save("temp_demo.jpg")
            else:
                st.image(img_file, caption="Image Originale", use_container_width=True)
    
    with col2:
        st.subheader("Analyse Visuelle")
        if img_file:
            if st.button("Lancer Analyse Image", type="primary"):
                with st.spinner("Analyse par le Conseil des Sages..."):
                    res = clf.predict_image("temp_demo.jpg")
                    
                    # explicabilite architecture
                    with st.expander("Comprendre le Vote (Architecture)", expanded=True):
                        st.write("Le resultat est un consensus pondere entre 3 modeles :")
                        k1, k2, k3 = st.columns(3)
                        k1.metric("DINOv3", "Poids 4", "Vision Globale")
                        k2.metric("EffNet", "Poids 2", "Details")
                        k3.metric("XGBoost", "Poids 1", "Stats")
                        
                        st.info("Note : XGBoost utilise une fonction de Sharpening (x^3) pour accentuer ses decisions.")
                    
                    show_results(res)

# ==========================================
# onglet 3 : fusion
# ==========================================
with tabs[2]:
    st.subheader("Cockpit de Fusion")
    st.info("La Fusion calcule une moyenne ponderee des probabilites issues du Texte et de l'Image.")
    
    # slider
    w_slider = st.slider("Reglage de la Balance (Poids Image)", 0.0, 1.0, 0.6)
    clf.w_image = w_slider
    clf.w_text = 1.0 - w_slider
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        f_txt = st.text_area("Texte", height=100, key="fus_t")
        f_img = st.file_uploader("Image", type=['jpg', 'png'], key="fus_i")
        
        go_fus = st.button("Calculer la Fusion", type="primary")
        
    with c2:
        if go_fus and f_txt and f_img:
            # preparation image
            with open("temp_fus.jpg", "wb") as f: f.write(f_img.getbuffer())
            
            with st.spinner("Synchronisation des modeles..."):
                # recuperation scores bruts pour explication
                r_txt = clf.predict_text(f_txt)[0]
                r_img = clf.predict_image("temp_fus.jpg")[0]
                
                # prediction finale
                final = clf.predict_fusion(f_txt, "temp_fus.jpg")
                
                # explicabilite mathematique
                with st.expander("Voir la formule mathematique", expanded=True):
                    st.latex(r"Score_{final} = (Score_{txt} \times Poids_{txt}) + (Score_{img} \times Poids_{img})")
                    
                    # calcul reel
                    sc_t = r_txt['confidence']
                    sc_i = r_img['confidence']
                    res_math = (sc_t * clf.w_text) + (sc_i * clf.w_image)
                    
                    st.code(f"{res_math:.4f} = ({sc_t:.4f} * {clf.w_text:.2f}) + ({sc_i:.4f} * {clf.w_image:.2f})")
                    
                    # analyse de coherence
                    if r_txt['label'] == r_img['label']:
                        st.success("Accord Parfait : Les deux modeles voient la meme chose.")
                    else:
                        st.warning(f"Conflit : Texte voit {r_txt['label']} | Image voit {r_img['label']}")
                
                show_results(final)