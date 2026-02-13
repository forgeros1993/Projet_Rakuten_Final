from pathlib import Path

# --- RACINE DU PROJET ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- ARBORESCENCE DONNÉES (Vital pour page 1_Données) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # <--- C'était l'erreur manquante
IMAGES_DIR = RAW_DATA_DIR / "images" / "images" / "image_train"

# --- ARBORESCENCE MODÈLES ---
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "implementation" / "outputs"

# --- ARBORESCENCE STREAMLIT ---
STREAMLIT_DIR = Path(__file__).parent
ASSETS_DIR = STREAMLIT_DIR / "assets"

# --- CHEMINS MODÈLES (LES PERFORMANTS) ---
# 1. DINOv3
IMAGE_MODEL_PATH = OUTPUTS_DIR / "model_DINOv3_BEST.pth" 
if not IMAGE_MODEL_PATH.exists():
    IMAGE_MODEL_PATH = MODELS_DIR / "M1_IMAGE_DeepLearning_DINOv3.pth"

# 2. XGBoost (Le fichier json + l'encodeur pkl)
XGB_MODEL_PATH = OUTPUTS_DIR / "M2_IMAGE_Classic_XGBoost.json"
XGB_ENC_PATH = OUTPUTS_DIR / "M2_IMAGE_XGBoost_Encoder.pkl"

# 3. EfficientNet
EFF_MODEL_PATH = OUTPUTS_DIR / "M3_IMAGE_Classic_EfficientNetB0.pth"

# 4. Texte & Mapping
TEXT_MODEL_PATH = MODELS_DIR / "text_classifier.joblib"
# On essaie le tfidf dans models, sinon implementation/outputs
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib" 
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.json"

# --- CONFIG APPLICATION ---
APP_CONFIG = {
    "title": "Rakuten Product Classifier",
    "icon": "🛒", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# --- CONFIG MODÈLE ---
MODEL_CONFIG = {
    "use_mock": False,
    # Poids Fusion : 30% Texte / 70% Image
    "fusion_weights_global": {"text": 0.5, "image": 0.5},
    # Poids Voting Image : DINO(4) / XGB(2) / Eff(1)
    "voting_weights": {"dino": 4.0, "xgb": 2.0, "effnet": 1.0}
}