import torch
import timm
import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from image_models.preprocessing import get_preprocessing, load_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictorM1:
    def __init__(self, model_path):
        self.model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=27)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        self.transform = get_preprocessing('M1')

    def predict_proba(self, image_path):
        """Renvoie le vecteur de probabilités (27,)"""
        img = load_image(image_path)
        if img is None: return None
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(tensor)
            # Softmax pour transformer les scores bruts en %
            probs = F.softmax(output, dim=1)
            return probs.cpu().numpy()[0]

    def predict(self, image_path):
        """Renvoie la classe finale (pour compatibilité)"""
        probs = self.predict_proba(image_path)
        if probs is None: return None
        return int(probs.argmax())