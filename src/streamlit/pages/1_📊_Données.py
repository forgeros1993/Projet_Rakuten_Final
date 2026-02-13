"""
Page d'exploration des données Rakuten.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.category_mapping import CATEGORY_MAPPING, get_category_info
from utils.data_loader import (
    is_data_available,
    get_category_distribution,
    get_text_statistics,
    get_sample_products,
    get_dataset_summary,
    load_training_data
)
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Données - {APP_CONFIG['title']}",
    page_icon="📊",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Exploration des Données")

if is_data_available():
    st.success("Données réelles chargées")
else:
    st.info("Mode démonstration")

# Métriques
st.divider()
summary = get_dataset_summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train", f"{summary['train_samples']:,}".replace(",", " "))
col2.metric("Test", f"{summary['test_samples']:,}".replace(",", " ") if isinstance(summary['test_samples'], int) else summary['test_samples'])
col3.metric("Catégories", summary['num_categories'])
col4.metric("Features", len(summary['features']))

# Distribution des catégories
st.divider()
st.header("Distribution des Catégories")

dist_df = get_category_distribution()

tab_bar, tab_table = st.tabs(["Graphique", "Tableau"])

with tab_bar:
    fig_bar = px.bar(
        dist_df,
        x='count',
        y='category_name',
        orientation='h',
        color='count',
        color_continuous_scale=['#FFE5E5', '#BF0000'],
        labels={'count': 'Produits', 'category_name': 'Catégorie'},
        text='count'
    )
    fig_bar.update_layout(
        height=700,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_table:
    display_df = dist_df[['emoji', 'category_name', 'count', 'percentage']].copy()
    display_df.columns = ['', 'Catégorie', 'Produits', '%']
    display_df['Produits'] = display_df['Produits'].apply(lambda x: f"{x:,}".replace(",", " "))
    display_df['%'] = display_df['%'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

# Statistiques texte
st.divider()
st.header("Statistiques Texte")

text_stats = get_text_statistics()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Désignation")
    desg = text_stats['designation']
    st.markdown(f"- Moyenne: **{desg['mean_length']:.0f}** car.")
    st.markdown(f"- Min/Max: **{desg['min_length']}** / **{desg['max_length']}**")

with col2:
    st.subheader("Description")
    desc = text_stats['description']
    st.markdown(f"- Moyenne: **{desc['mean_length']:.0f}** car.")
    st.markdown(f"- Remplissage: **{desc['non_empty_pct']:.0f}%**")

# Exemples
st.divider()
st.header("Exemples par Catégorie")

categories_list = [(f"{info[2]} {info[0]}", code) for code, info in CATEGORY_MAPPING.items()]
selected = st.selectbox("Catégorie", categories_list, format_func=lambda x: x[0])

X_train, Y_train = load_training_data()
samples = get_sample_products(X_train, Y_train, category_code=selected[1], n_samples=3)

if len(samples) > 0:
    for _, row in samples.iterrows():
        with st.expander(f"{row['designation'][:60]}..."):
            st.write(f"**Désignation:** {row['designation']}")
            desc = row.get('description', '')
            if pd.notna(desc) and str(desc).strip():
                st.write(f"**Description:** {str(desc)[:300]}...")

# Déséquilibre
st.divider()
st.header("Déséquilibre des Classes")

col1, col2, col3 = st.columns(3)
col1.metric("Majoritaire", dist_df.iloc[0]['category_name'], f"{dist_df.iloc[0]['count']:,}".replace(",", " "))
col2.metric("Minoritaire", dist_df.iloc[-1]['category_name'], f"{dist_df.iloc[-1]['count']:,}".replace(",", " "))
col3.metric("Ratio", f"{dist_df['count'].max() / dist_df['count'].min():.1f}x")

st.info("Le déséquilibre est géré par class weighting et SMOTE.")

# Sidebar
with st.sidebar:
    st.markdown("### Données")
    st.divider()
    csv_data = dist_df.to_csv(index=False)
    st.download_button("Télécharger CSV", csv_data, "distribution.csv", "text/csv")
