# src/features/build_features.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# je definis les constantes globales
IMG_SIZE_DINO = 518
IMG_SIZE_EFF = 224

# je prepare la transformation pour dino
def get_dino_transforms():
    # augmentation specifique dino
    # resize bicubic + normalisation imagenet
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE_DINO, IMG_SIZE_DINO), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# je prepare la transformation pour efficientnet
def get_efficientnet_transforms():
    # resize standard 224
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE_EFF, IMG_SIZE_EFF)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# je charge une image depuis le disque
def load_and_process_image(image_path, model_type='dino'):
    try:
        # lecture opencv
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # choix transform
        if model_type == 'dino':
            trans = get_dino_transforms()
        else:
            trans = get_efficientnet_transforms()
            
        # application
        tensor = trans(img)
        # ajout dimension batch (1, 3, h, w)
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"err lecture img: {e}")
        return None