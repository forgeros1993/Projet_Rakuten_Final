from pathlib import Path
import os

# ==============================================================================
# 🌍 CONFIGURATION SPÉCIALE GOOGLE DRIVE
# ==============================================================================

# 1. DÉFINITION DE LA RACINE DU PROJET (DRIVE G:)
# On vise spécifiquement ton dossier de sauvegarde
DRIVE_ROOT = Path("G:/Mon Drive/travail final/FULL_BACKUP_RAPIDE/datascientest_projet")

# Sécurité : Si le Drive n'est pas monté, on essaie de trouver le chemin relatif
if DRIVE_ROOT.exists():
    PROJECT_ROOT = DRIVE_ROOT
    print(f"✅ CONFIG: Racine trouvée sur le Drive G: {PROJECT_ROOT}")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print(f"⚠️ CONFIG: Drive introuvable, utilisation du chemin relatif : {PROJECT_ROOT}")

# --- ARBORESCENCE DONNÉES ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 2. CHEMIN DES IMAGES (Vital pour l'affichage)
# On pointe vers le dossier image_train dans le backup
IMAGES_DIR = RAW_DATA_DIR / "images" / "images" / "image_train"

# --- ARBORESCENCE MODÈLES ---
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "implementation" / "outputs"

# --- ARBORESCENCE STREAMLIT ---
STREAMLIT_DIR = Path(__file__).parent
ASSETS_DIR = STREAMLIT_DIR / "assets"

# ==============================================================================
# 🧠 CHEMINS DES CERVEAUX (MODÈLES)
# ==============================================================================

# 1. DINOv3 (Le Vision Transformer)
IMAGE_MODEL_PATH = OUTPUTS_DIR / "model_DINOv3_BEST.pth" 
if not IMAGE_MODEL_PATH.exists():
    # Fallback si jamais il est rangé ailleurs
    IMAGE_MODEL_PATH = MODELS_DIR / "M1_IMAGE_DeepLearning_DINOv3.pth"

# 2. XGBoost TITAN (Json + Encoder)
XGB_MODEL_PATH = OUTPUTS_DIR / "M2_IMAGE_Classic_XGBoost.json"
XGB_ENC_PATH = OUTPUTS_DIR / "M2_IMAGE_XGBoost_Encoder.pkl"

# 3. EfficientNet
EFF_MODEL_PATH = OUTPUTS_DIR / "M3_IMAGE_Classic_EfficientNetB0.pth"

# 4. Texte (Joblib)
TEXT_MODEL_PATH = MODELS_DIR / "text_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib" 
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.json"

# --- CONFIG APPLICATION ---
APP_CONFIG = {
    "title": "Rakuten Product Classifier",
    "icon": "🛒", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# --- CONFIG MODÈLE (POIDS) ---
MODEL_CONFIG = {
    "use_mock": False,
    # Poids Fusion : 50% Texte / 50% Image (Équilibré)
    "fusion_weights_global": {"text": 0.5, "image": 0.5},
    # Poids Voting Image : DINO(4) / XGB(2) / Eff(1)
    "voting_weights": {"dino": 4.0, "xgb": 2.0, "effnet": 1.0}
}