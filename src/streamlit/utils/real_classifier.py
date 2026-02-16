import sys
import os
import joblib
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import xgboost as xgb
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

# --- BLOC KERAS (OBLIGATOIRE POUR LE TITAN) ---
# On charge TensorFlow pour que le XGBoost reçoive les bonnes features
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
# ----------------------------------------------

# Import config sécurisé
try:
    from config import (
        IMAGE_MODEL_PATH, XGB_MODEL_PATH, EFF_MODEL_PATH,
        TEXT_MODEL_PATH, CATEGORY_MAPPING_PATH, MODEL_CONFIG
    )
except ImportError:
    pass

class MultimodalClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initialisation MultimodalClassifier sur {self.device}")
        
        # 1. LISTE OFFICIELLE DES CODES RAKUTEN
        self.OFFICIAL_CODES = [
            "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
            "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
            "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
            "2705", "2905"
        ]
        
        # 2. Chargement Mapping (EN DUR POUR AFFICHAGE PROPRE)
        self.mapping = self._load_hardcoded_mapping()
        
        # 3. Chargement Modèles Images
        self.img_models = self._load_image_models()
        self.img_transforms = self._init_transforms()
        
        # 4. Chargement Modèle Texte
        self.text_model = self._load_text_model()

    def _load_hardcoded_mapping(self):
        """
        Dictionnaire complet pour remplacer les codes par du texte lisible.
        C'est ce qui s'affichera dans la démo.
        """
        return {
            "10": "Livres occasion",
            "40": "Jeux Vidéo",
            "50": "Accessoires Gaming",
            "60": "Consoles de jeux",
            "1140": "Figurines & Pop",
            "1160": "Cartes (Pokémon/Magic)",
            "1180": "Jeux de Rôle & Figurines",
            "1280": "Jouets Enfants / Peluches",
            "1281": "Jeux de Société",
            "1300": "Modélisme (Drones/RC)",
            "1301": "Loisirs Créatifs / Beaux-Arts",
            "1302": "Fournitures Scolaires",
            "1320": "Puériculture & Bébés",
            "1560": "Mobilier & Déco",
            "1920": "Linge de Maison",
            "1940": "Épicerie & Confitiserie",
            "2060": "Décoration Intérieure",
            "2220": "Animalerie",
            "2280": "Journaux & Magazines",
            "2403": "BD, Mangas & Comics",
            "2462": "Jeux Vidéo (Dématérialisé)",
            "2522": "Papeterie & Bureau",
            "2582": "Jardin & Extérieur",
            "2583": "Piscine & Spa",
            "2585": "Bricolage & Outillage",
            "2705": "Informatique & High-Tech",
            "2905": "Jeux PC (Codes)"
        }

    def _load_text_model(self):
        try:
            if os.path.exists(TEXT_MODEL_PATH):
                return joblib.load(TEXT_MODEL_PATH)
        except Exception: pass
        return None

    def _load_image_models(self):
        loaded = {}
        
        # --- A. DINOv3 (PyTorch) ---
        try:
            # On tente de charger Reg4 sinon standard
            try:
                dino = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=27)
            except:
                dino = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, num_classes=27)
                
            if os.path.exists(IMAGE_MODEL_PATH):
                dino.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=self.device))
                dino.to(self.device).eval()
                loaded['dino'] = dino
                print("✅ DINOv3 chargé")
        except Exception as e: print(f"❌ Erreur DINO: {e}")

        # --- B. EfficientNet (PyTorch) ---
        try:
            eff = models.efficientnet_b0(weights=None)
            eff.classifier[1] = nn.Linear(1280, 27)
            if os.path.exists(EFF_MODEL_PATH):
                eff.load_state_dict(torch.load(EFF_MODEL_PATH, map_location=self.device))
                eff.to(self.device).eval()
                loaded['effnet'] = eff
                print("✅ EfficientNet chargé")
        except Exception as e: print(f"❌ Erreur EffNet: {e}")

        # --- C. XGBoost (PIPELINE KERAS HYBRIDE) ---
        try:
            # On force la conversion en string pour XGBoost car il n'aime pas les objets Path de Windows
            xgb_path_str = str(XGB_MODEL_PATH)
            
            if os.path.exists(xgb_path_str):
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(xgb_path_str)
                loaded['xgb'] = xgb_model
                
                print("⏳ Chargement Keras ResNet50 pour Titan...")
                self.keras_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                print("✅ XGBoost + Keras ResNet chargés")
            else:
                print(f"⚠️ XGBoost introuvable à : {xgb_path_str}")
        except Exception as e: print(f"❌ Erreur XGBoost Pipeline: {e}")
        
        return loaded

    def _init_transforms(self):
        return {
            'dino': transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'effnet': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def predict_image(self, image_path):
        try:
            prob_dino = np.zeros(27)
            prob_eff = np.zeros(27)
            prob_xgb = np.zeros(27)

            # 1. DINO (PyTorch)
            if 'dino' in self.img_models:
                img_pil = Image.open(image_path).convert('RGB')
                t_dino = self.img_transforms['dino'](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob_dino = F.softmax(self.img_models['dino'](t_dino), dim=1).cpu().numpy()[0]

            # 2. EfficientNet (PyTorch)
            if 'effnet' in self.img_models:
                img_pil = Image.open(image_path).convert('RGB')
                t_eff = self.img_transforms['effnet'](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob_eff = F.softmax(self.img_models['effnet'](t_eff), dim=1).cpu().numpy()[0]

            # 3. XGBoost (Keras Preprocessing)
            if 'xgb' in self.img_models and hasattr(self, 'keras_resnet'):
                # Force path en string pour Keras
                img_k = keras_image.load_img(str(image_path), target_size=(224, 224))
                x = keras_image.img_to_array(img_k)
                x = preprocess_input(x)
                x = np.expand_dims(x, axis=0)
                feats = self.keras_resnet.predict(x, verbose=0)
                prob_xgb = self.img_models['xgb'].predict_proba(feats)[0]

            # 4. Voting
            w = MODEL_CONFIG['voting_weights']
            final_prob = np.zeros(27)
            total_weight = 0.0

            if 'dino' in self.img_models:
                final_prob += prob_dino * w['dino']
                total_weight += w['dino']
            if 'xgb' in self.img_models:
                final_prob += prob_xgb * w['xgb']
                total_weight += w['xgb']
            if 'effnet' in self.img_models:
                final_prob += prob_eff * w['effnet']
                total_weight += w['effnet']

            if total_weight > 0: final_prob /= total_weight
            else: final_prob[0] = 1.0 

            return self._format_output(final_prob)

        except Exception as e:
            print(f"❌ CRASH IMAGE: {e}")
            return self._fallback_result()

    def predict_text(self, text):
        if not self.text_model: return self._fallback_result()
        try:
            t = [text] if isinstance(text, str) else text
            if hasattr(self.text_model, "predict_proba"):
                probs = self.text_model.predict_proba(t)[0]
            else:
                probs = np.zeros(27); probs[0] = 1.0 
            return self._format_output(probs)
        except Exception: return self._fallback_result()

    def predict_fusion(self, text, image_path):
        try:
            r_txt = self.predict_text(text)
            r_img = self.predict_image(image_path)
            s_txt = {r['label']: r['confidence'] for r in r_txt}
            s_img = {r['label']: r['confidence'] for r in r_img}
            w_t = MODEL_CONFIG['fusion_weights_global']['text']
            w_i = MODEL_CONFIG['fusion_weights_global']['image']
            
            final_probs = np.zeros(27)
            for i, code in enumerate(self.OFFICIAL_CODES):
                score = (s_txt.get(code, 0)*w_t) + (s_img.get(code, 0)*w_i)
                final_probs[i] = score
            
            return self._format_output(final_probs)
        except Exception: return self._fallback_result()

    def _format_output(self, probabilities):
        results = []
        for i, p in enumerate(probabilities):
            if i < len(self.OFFICIAL_CODES):
                real_code = self.OFFICIAL_CODES[i]
                # ICI : ON UTILISE LE MAPPING EN DUR
                name = self.mapping.get(real_code, f"Code {real_code}")
                results.append({"label": real_code, "name": name, "confidence": float(p)})
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _fallback_result(self):
        return [
            {"label": "10", "name": "Livres (Mode Secours)", "confidence": 0.8},
            {"label": "40", "name": "Jeux Vidéo (Mode Secours)", "confidence": 0.2}
        ]