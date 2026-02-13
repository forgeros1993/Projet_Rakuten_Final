import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from image_models.preprocessing import get_preprocessing, load_image

DEVICE = torch.device("cpu") # M3 sur CPU pour alléger la VRAM du Voting

class PredictorM3:
    def __init__(self, model_path):
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, 27)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        self.transform = get_preprocessing('M3')

    def predict_proba(self, image_path):
        """Renvoie le vecteur de probabilités (27,)"""
        img = load_image(image_path)
        if img is None: return None
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            return probs.cpu().numpy()[0]

    def predict(self, image_path):
        probs = self.predict_proba(image_path)
        if probs is None: return None
        return int(probs.argmax())