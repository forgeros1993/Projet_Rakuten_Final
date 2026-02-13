import xgboost as xgb
import joblib
import torch
import timm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from image_models.preprocessing import get_preprocessing, load_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictorM2:
    def __init__(self, xgb_path, encoder_path):
        # Extracteur de features (ResNet50)
        self.extractor = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.extractor.to(DEVICE).eval()
        self.transform = get_preprocessing('M3') # M3/M4 utilisent la transfo standard 224
        
        # XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_path)
        self.le = joblib.load(encoder_path)

    def predict_proba(self, image_path):
        """Renvoie le vecteur de probabilités (27,)"""
        img = load_image(image_path)
        if img is None: return None
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = self.extractor(tensor).cpu().numpy()
        
        # XGBoost renvoie directement les probas
        probs = self.xgb_model.predict_proba(features)
        return probs[0]

    def predict(self, image_path):
        """Renvoie la classe finale décodée"""
        probs = self.predict_proba(image_path)
        if probs is None: return None
        # On récupère l'index max
        pred_idx = probs.argmax()
        # On décode avec le LabelEncoder pour avoir le vrai ProductCode
        return self.le.inverse_transform([pred_idx])[0]