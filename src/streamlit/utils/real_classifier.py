import sys
import os
import joblib
import json
import numpy as np
import cv2
import torch
import timm
import xgboost as xgb
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

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
        
        # 1. LISTE OFFICIELLE DES CODES RAKUTEN (ORDRE TRIÉ STANDARD)
        # C'est l'ordre utilisé par XGBoost et Scikit-Learn par défaut.
        # On s'en sert pour traduire l'index 0 -> "10", index 1 -> "40", etc.
        self.OFFICIAL_CODES = [
            "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
            "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
            "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
            "2705", "2905"
        ]
        
        # 2. Chargement Mapping (Code -> Nom)
        self.mapping = self._load_mapping()
        
        # 3. Chargement Modèles Images
        self.img_models = self._load_image_models()
        self.img_transforms = self._init_transforms()
        
        # 4. Chargement Modèle Texte
        self.text_model = self._load_text_model()

    def _load_mapping(self):
        """Charge le mapping avec double clés (int et str) pour sécurité absolue"""
        mapping = {}
        
        # A. Tentative JSON
        if os.path.exists(CATEGORY_MAPPING_PATH):
            try:
                with open(CATEGORY_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    raw_map = json.load(f)
                    for k, v in raw_map.items():
                        mapping[str(k).strip()] = v  # Clé string "10"
                        try: mapping[int(k)] = v     # Clé int 10
                        except: pass
            except Exception: pass
        
        # B. Dictionnaire de secours (Si JSON vide)
        if len(mapping) < 10:
            print("⚠️ Mapping JSON non trouvé/vide. Utilisation du mapping de secours.")
            raw_fallback = {
                10: "Livres occasion", 40: "Jeux Vidéo", 50: "Accessoires Gaming",
                60: "Consoles", 1140: "Figurines", 1160: "Cartes à jouer",
                1180: "Jeux de Rôle", 1280: "Jouets enfants", 1281: "Jeux de société",
                1300: "Modélisme", 1301: "Loisirs créatifs", 1302: "Fournitures scolaires",
                1320: "Puériculture", 1560: "Meubles", 1920: "Linge de maison",
                1940: "Epicerie", 2060: "Décoration", 2220: "Animalerie",
                2280: "Magazines", 2403: "BD et Comics", 2462: "Jeux PC",
                2522: "Papeterie", 2582: "Jardin", 2583: "Piscine / Spa",
                2585: "Bricolage", 2705: "Informatique", 2905: "Jeux Vidéo (Dém.)"
            }
            for k, v in raw_fallback.items():
                mapping[str(k)] = v
                mapping[k] = v
                
        return mapping

    def _load_text_model(self):
        try:
            if os.path.exists(TEXT_MODEL_PATH):
                return joblib.load(TEXT_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Erreur chargement texte (Version mismatch possible): {e}")
        return None

    def _load_image_models(self):
        loaded = {}
        # DINOv3
        try:
            dino = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=27)
            if os.path.exists(IMAGE_MODEL_PATH):
                dino.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=self.device))
                dino.to(self.device).eval()
                loaded['dino'] = dino
                print("✅ DINOv3 chargé")
        except Exception: pass

        # EfficientNet
        try:
            eff = models.efficientnet_b0(weights=None)
            eff.classifier[1] = nn.Linear(1280, 27)
            if os.path.exists(EFF_MODEL_PATH):
                eff.load_state_dict(torch.load(EFF_MODEL_PATH, map_location=self.device))
                eff.to(self.device).eval()
                loaded['effnet'] = eff
                print("✅ EfficientNet chargé")
        except Exception: pass

        # XGBoost + ResNet
        try:
            if os.path.exists(XGB_MODEL_PATH):
                resnet = models.resnet50(weights="IMAGENET1K_V1")
                extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
                loaded['extractor'] = extractor
                
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(str(XGB_MODEL_PATH))
                loaded['xgb'] = xgb_model
                print("✅ XGBoost chargé")
                # ON NE CHARGE PLUS L'ENCODEUR .PKL ICI CAR IL EST BUGGÉ
                # On utilisera self.OFFICIAL_CODES à la place
        except Exception as e: print(f"❌ Erreur XGBoost: {e}")
        
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
        """Pipeline Image Hybride (PIL + OpenCV)"""
        try:
            prob_dino = np.zeros(27)
            prob_eff = np.zeros(27)
            prob_xgb = np.zeros(27)

            # 1. DINO
            if 'dino' in self.img_models:
                img_pil = Image.open(image_path).convert('RGB')
                t_dino = self.img_transforms['dino'](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob_dino = F.softmax(self.img_models['dino'](t_dino), dim=1).cpu().numpy()[0]

            # 2. EffNet
            if 'effnet' in self.img_models:
                img_pil = Image.open(image_path).convert('RGB')
                t_eff = self.img_transforms['effnet'](img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob_eff = F.softmax(self.img_models['effnet'](t_eff), dim=1).cpu().numpy()[0]

            # 3. XGBoost (OpenCV STRICT)
            if 'xgb' in self.img_models:
                img_cv = cv2.imread(str(image_path))
                if img_cv is not None:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    img_cv = cv2.resize(img_cv, (224, 224))
                    img_cv = img_cv / 255.0
                    img_cv = (img_cv - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    
                    t_xgb = torch.tensor(img_cv.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feats = self.img_models['extractor'](t_xgb).squeeze().cpu().numpy()
                    prob_xgb = self.img_models['xgb'].predict_proba(feats.reshape(1, -1))[0]

            # 4. Voting
            w = MODEL_CONFIG['voting_weights']
            final_prob = (w['dino']*prob_dino + w['xgb']*prob_xgb + w['effnet']*prob_eff) / sum(w.values())

            return self._format_output(final_prob)

        except Exception as e:
            print(f"❌ CRASH IMAGE: {e}")
            return self._fallback_result()

    def predict_text(self, text):
        if not self.text_model: return self._fallback_result()
        try:
            t = [text] if isinstance(text, str) else text
            
            # Gestion modèles sans predict_proba (LinearSVC)
            if hasattr(self.text_model, "predict_proba"):
                probs = self.text_model.predict_proba(t)[0]
            elif hasattr(self.text_model, "decision_function"):
                s = self.text_model.decision_function(t)[0]
                probs = np.exp(s - np.max(s))
                probs /= probs.sum()
            else:
                probs = np.zeros(27); probs[0] = 1.0 
            
            # Formatage
            results = []
            # On utilise self.OFFICIAL_CODES car le modèle texte a été entrainé sur les mêmes labels
            for i, p in enumerate(probs):
                code = self.OFFICIAL_CODES[i] if i < len(self.OFFICIAL_CODES) else str(i)
                results.append({
                    "label": code,
                    "confidence": float(p),
                    "name": self.mapping.get(code, self.mapping.get(int(code), "Inconnu"))
                })
            return sorted(results, key=lambda x: x['confidence'], reverse=True)
            
        except Exception: return self._fallback_result()

    def predict_fusion(self, text, image_path):
        try:
            r_txt = self.predict_text(text)
            r_img = self.predict_image(image_path)
            
            s_txt = {r['label']: r['confidence'] for r in r_txt}
            s_img = {r['label']: r['confidence'] for r in r_img}
            
            w_t = MODEL_CONFIG['fusion_weights_global']['text']
            w_i = MODEL_CONFIG['fusion_weights_global']['image']
            
            final = []
            for lbl in set(s_txt) | set(s_img):
                score = (s_txt.get(lbl, 0)*w_t) + (s_img.get(lbl, 0)*w_i)
                final.append({
                    "label": lbl, 
                    "confidence": score,
                    "name": self.mapping.get(lbl, self.mapping.get(int(lbl) if lbl.isdigit() else lbl, "Inconnu"))
                })
            return sorted(final, key=lambda x: x['confidence'], reverse=True)
        except Exception: return self._fallback_result()

    def _format_output(self, probabilities):
        results = []
        for i, p in enumerate(probabilities):
            # ON N'UTILISE PLUS LE LE.INVERSE_TRANSFORM CORROMPU
            # On mappe directement l'index i sur la liste triée officielle
            real_code = self.OFFICIAL_CODES[i] if i < len(self.OFFICIAL_CODES) else str(i)

            # On cherche le nom
            # On essaie en str ("10") puis en int (10)
            name = self.mapping.get(real_code, self.mapping.get(int(real_code) if real_code.isdigit() else -1, "Inconnu"))
            
            results.append({"label": real_code, "name": name, "confidence": float(p)})
            
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _fallback_result(self):
        return [
            {"label": "10", "name": "Livres (Mode Secours)", "confidence": 0.8},
            {"label": "40", "name": "Jeux Vidéo (Mode Secours)", "confidence": 0.2}
        ]