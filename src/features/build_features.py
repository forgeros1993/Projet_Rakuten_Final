import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(img_path, mode="standard"):
    # nous chargeons image
    img = Image.open(img_path).convert('RGB')
    if mode == "dino":
        # dino veut du 518x518
        tr = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # standard reste a 224
        tr = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return tr(img).unsqueeze(0)
