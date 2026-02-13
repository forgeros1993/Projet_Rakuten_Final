import cv2
import torch
from torchvision import transforms

# TAILLES
SIZE_DINO = 518
SIZE_STD = 224

def get_preprocessing(model_id):
    if model_id == 'M1': # DINO
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((SIZE_DINO, SIZE_DINO), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif model_id in ['M3', 'M4']: # EffNet et ResNet Overfit
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((SIZE_STD, SIZE_STD)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else: raise ValueError(f"ID inconnu {model_id}")

def load_image(path):
    try:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except: return None