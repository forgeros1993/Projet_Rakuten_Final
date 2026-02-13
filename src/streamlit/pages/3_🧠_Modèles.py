import streamlit as st
import pandas as pd
import graphviz
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
from pathlib import Path
import scipy.stats as stats

# --- GESTION DES CHEMINS ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASSETS_DIR
from utils.ui_utils import load_css

# --- CONFIG PAGE ---
st.set_page_config(page_title="Architecture", layout="wide")
load_css(ASSETS_DIR / "style.css")

st.title("🧠 Architecture & Performance")
st.markdown("---")

tabs = st.tabs(["🖼️ Vision (Image)", "📝 Sémantique (Texte)"])

# ==========================================
# ONGLET 1 : VISION (IMAGE)
# ==========================================
with tabs[0]:
    # --- ZONE 1 : LE PIPELINE (Graphviz Optimisé 16/9) ---
    st.markdown("#### 1. Pipeline de Décision : Le 'Voting'")
    
    try:
        graph = graphviz.Digraph()
        # rankdir='LR' = Horizontal (parfait pour écran large)
        # size='10,4' limite la taille maximale
        graph.attr(rankdir='LR', bgcolor='white', ranksep='0.5', nodesep='0.3', size='10,3')
        
        # 1. Entrée et Preprocessing
        graph.node('I', 'Input\n(Image)', shape='oval', style='filled', fillcolor='#e0e0e0', fontsize='11')
        graph.node('P', 'Preprocessing\n(Resize/Norm)', shape='box', style='rounded,filled', fillcolor='#fff9c4', fontsize='11')

        # 2. Les Modèles (Alignés verticalement via subgraph)
        with graph.subgraph() as s:
            s.attr(rank='same')
            s.node('D', 'DINOv3\n(Transformer)', style='filled', fillcolor='#d1c4e9', fontsize='11')
            s.node('E', 'EffNet\n(CNN)', style='filled', fillcolor='#b3e5fc', fontsize='11')
            s.node('X', 'XGBoost\n(Tabular)', style='filled', fillcolor='#c8e6c9', fontsize='11')

        # 3. Sortie
        graph.node('V', 'VOTING\n(Fusion)', shape='Mdiamond', style='filled', fillcolor='#ff8a80', fontsize='12')
        graph.node('O', 'Sortie', shape='doublecircle', style='filled', fillcolor='gold', fontsize='11')
        
        # 4. Connexions
        graph.edge('I', 'P')
        graph.edge('P', 'D')
        graph.edge('P', 'E')
        
        graph.edge('D', 'X', style='dashed', color='gray')
        graph.edge('E', 'X', style='dashed', color='gray')

        graph.edge('D', 'V', label='x4', color='#7e57c2', penwidth='2')
        graph.edge('E', 'V', label='x2', color='#039be5', penwidth='2')
        graph.edge('X', 'V', label='x1', color='#2e7d32', penwidth='1.5')
        
        graph.edge('V', 'O')
        
        # Centrage du graphe
        c_Left, c_Center, c_Right = st.columns([1, 8, 1])
        with c_Center:
            st.graphviz_chart(graph, use_container_width=True)
            
    except Exception as e:
        st.info("Pipeline : DINOv2 (x4) + EfficientNet (x2) + XGBoost (x1) -> Vote Majoritaire")
    
    st.markdown("---")
    
    # --- ZONE 2 : DASHBOARD INTERACTIF ---
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    
    # --- GRAPHIQUE 1 : RADAR CHART ---
    with c1:
        st.markdown("##### 🎯 Profils (Radar)")
        categories = ['Vitesse', 'Précision', 'Confiance', 'Robustesse', 'Universalité']
        fig_radar = go.Figure()

        # XGBoost 
        fig_radar.add_trace(go.Scatterpolar(
            r=[9, 6, 4, 5, 4], theta=categories, fill='toself', name='XGBoost',
            line=dict(color='green'), opacity=0.4
        ))
        # DINOv3
        fig_radar.add_trace(go.Scatterpolar(
            r=[3, 8, 8, 9, 9], theta=categories, fill='toself', name='DINOv3',
            line=dict(color='blue'), opacity=0.4
        ))
        # VOTING
        fig_radar.add_trace(go.Scatterpolar(
            r=[6, 9.5, 9.5, 10, 9.5], theta=categories, fill='toself', name='VOTING',
            line=dict(color='#BF0000', width=3), opacity=0.7
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True, legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=20, b=20), height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("✅ **Analyse :** Le Voting (Rouge) cumule les forces.")

    # --- GRAPHIQUE 2 : CALIBRATION ---
    with c2:
        st.markdown("##### 📐 Calibration")
        x = np.linspace(0, 1.1, 200)
        y_xgb = stats.norm.pdf(x, 0.3, 0.12)
        y_vot = stats.norm.pdf(x, 0.88, 0.08)
        
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(x=x, y=y_xgb, fill='tozeroy', mode='none', name='XGBoost', fillcolor='rgba(0, 128, 0, 0.3)'))
        fig_calib.add_trace(go.Scatter(x=x, y=y_vot, fill='tozeroy', mode='none', name='VOTING', fillcolor='rgba(191, 0, 0, 0.5)'))
        
        fig_calib.add_vline(x=0.8, line_width=2, line_dash="dash", line_color="red")
        fig_calib.add_annotation(x=0.85, y=3, text="Seuil Auto (80%)", textangle=-90, showarrow=False, font=dict(color="red"))

        fig_calib.update_layout(xaxis_title="Confiance", yaxis_visible=False, showlegend=True, legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=20, b=20), height=300)
        st.plotly_chart(fig_calib, use_container_width=True)
        st.caption("✅ **Gain :** Le Voting passe le seuil d'automatisation.")

    # --- GRAPHIQUE 3 : MATRICE ---
    with c3:
        st.markdown("##### 🤝 Diversité")
        matrix_z = [[100, 56, 56, 0, 56], [56, 100, 62, 0, 62], [56, 62, 100, 0, 100], [0, 0, 0, 100, 0], [56, 62, 100, 0, 100]]
        labels = ["Phoen.", "EffNet", "DINO", "XGB", "Vote"]
        annotations = [["100%", "56%", "56%", "Indép.", "56%"], ["56%", "100%", "62%", "Indép.", "62%"], ["56%", "62%", "100%", "Indép.", "100%"], ["Indép.", "Indép.", "Indép.", "100%", "6%"], ["56%", "62%", "100%", "Indép.", "100%"]]

        fig_mat = go.Figure(data=go.Heatmap(z=matrix_z, x=labels, y=labels, text=annotations, texttemplate="%{text}", colorscale='Blues', showscale=False))
        fig_mat.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption("✅ **Stratégie :** XGBoost sécurise le vote.")

# ==========================================
# ONGLET 2 : SEMANTIQUE (TEXTE)
# ==========================================
with tabs[1]:
    st.subheader("Traitement du Langage Naturel (NLP)")
    
    col_txt1, col_txt2 = st.columns([1, 1], gap="large")
    
    with col_txt1:
        st.markdown("#### Pipeline Technique")
        
        # --- CORRECTION VISUELLE : MODE HORIZONTAL COMPACT ---
        try:
            graph_txt = graphviz.Digraph()
            # rankdir='LR' force l'horizontalité (gauche -> droite)
            # size='8,2' force une petite hauteur pour éviter l'effet "Géant"
            graph_txt.attr(rankdir='LR', bgcolor='white', margin='0.1', size='8,2', ranksep='0.4')
            
            # Noeuds simplifiés et esthétiques
            graph_txt.node('1', 'Entrée\n(Texte)', shape='oval', style='filled', fillcolor='#f5f5f5', fontsize='11')
            graph_txt.node('2', 'Nettoyage\n(Regex/Clean)', shape='box', style='rounded,filled', fillcolor='#e1f5fe', fontsize='11')
            graph_txt.node('3', 'Vectorisation\n(TF-IDF)', shape='box', style='filled', fillcolor='#e1bee7', fontsize='11')
            graph_txt.node('4', 'Modèle\n(LinearSVC)', shape='ellipse', style='filled', fillcolor='#ffe0b2', fontsize='11')
            
            graph_txt.edge('1', '2')
            graph_txt.edge('2', '3')
            graph_txt.edge('3', '4')
            
            st.graphviz_chart(graph_txt, use_container_width=True)
            
        except:
            st.info("Pipeline : Texte -> Cleaning -> TF-IDF -> LinearSVC")
        
        st.info("⚡ **Vitesse :** < 10ms par produit. Idéal temps réel.")

    with col_txt2:
        st.markdown("#### Benchmark Texte")
        
        df_text = pd.DataFrame({
            "Modèle": ["LinearSVC", "Random Forest", "LogReg", "CamemBERT"],
            "F1-Score": ["84.1%", "72.0%", "69.5%", "81.0%"],
            "Vitesse": ["Fast", "Medium", "Fast", "Slow"]
        })
        
        st.dataframe(
            df_text.style.highlight_max(
                axis=0, 
                subset=["F1-Score"], 
                props="color: black; background-color: #d4edda; font-weight: bold;"
            ),
            use_container_width=True,
            hide_index=True
        )