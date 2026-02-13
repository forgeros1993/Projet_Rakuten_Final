import numpy as np
import os
import sys

# imports relatifs
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from image_models.predict.predict_M1_dino import PredictorM1
from image_models.predict.predict_M3_effnet import PredictorM3
# on n'importe pas m2 car il nécessite des features npy introuvables sur images brutes

class PredictorVoting:
    def __init__(self, dir_models):
        print("initialisation voting classifier (mode image : m1 + m3)...")
        
        # chemins
        path_m1 = os.path.join(dir_models, "M1_IMAGE_DeepLearning_DINOv3.pth")
        path_m3 = os.path.join(dir_models, "M3_IMAGE_Classic_EfficientNetB0.pth")
        
        # chargement
        self.p1 = PredictorM1(path_m1)
        self.p3 = PredictorM3(path_m3)
        
        # poids (dino majoritaire)
        self.w1 = 2.0
        self.w3 = 1.0

    def predict_proba(self, image_path):
        # recuperation probas
        probs_1 = self.p1.predict_proba(image_path)
        probs_3 = self.p3.predict_proba(image_path)
        
        if probs_1 is None or probs_3 is None:
            return None
            
        # moyenne ponderee
        final_probs = (probs_1 * self.w1 + probs_3 * self.w3) / (self.w1 + self.w3)
        return final_probs

    def predict(self, image_path):
        # on recupere les probas fusionnees
        probs = self.predict_proba(image_path)
        if probs is None: return None
        
        # decision finale
        final_idx = probs.argmax()
        
        # on utilise le decodeur de m3 (label encoder) pour renvoyer le vrai code produit
        # (les classes sont alignees entre m1 et m3 lors de l'entrainement)
        # ici on assume que p1 et p3 ont les memes classes (ce qui est le cas dans  notre projet)
        return self.p1.le.inverse_transform([final_idx])[0] # ou p3, c'est pareil

if __name__ == "__main__":
    print("voting m1+m3 prêt.")